#include "client.hpp"

#include <boost/asio/use_awaitable.hpp>
#include <boost/json.hpp>
#include <chrono>
#include <sstream>

namespace json = boost::json;
using tcp = net::ip::tcp;
using clock_type = std::chrono::steady_clock;

static double ms_between(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// True when an SSE data payload carries a non-empty content delta.
static bool has_content(const std::string& payload) {
    if (payload == "[DONE]") return false;
    std::error_code ec;
    auto value = json::parse(payload, ec);
    if (ec || !value.is_object()) return false;
    const auto& obj = value.as_object();
    auto* choices = obj.if_contains("choices");
    if (!choices || !choices->is_array() || choices->as_array().empty())
        return false;
    const auto& first = choices->as_array()[0];
    if (!first.is_object()) return false;
    const auto* delta = first.as_object().if_contains("delta");
    if (!delta || !delta->is_object()) return false;
    const auto* content = delta->as_object().if_contains("content");
    return content && content->is_string() && !content->as_string().empty();
}

static std::string build_body(const std::string& model,
                              const std::string& prompt, int max_tokens) {
    json::object body;
    body["model"] = model;
    json::array messages;
    json::object msg;
    msg["role"] = "user";
    msg["content"] = prompt;
    messages.push_back(msg);
    body["messages"] = messages;
    body["max_tokens"] = max_tokens;
    body["temperature"] = 0.0;
    body["stream"] = true;
    // Ask for a final usage block where the server supports it; harmless
    // where it does not.
    json::object opts;
    opts["include_usage"] = true;
    body["stream_options"] = opts;
    return json::serialize(body);
}

net::awaitable<RequestRecord> run_request(net::io_context& ioc,
                                          const std::string& host,
                                          const std::string& port,
                                          const std::string& path,
                                          const std::string& api_key,
                                          const std::string& model,
                                          const std::string& prompt,
                                          int prompt_tokens,
                                          int max_tokens) {
    RequestRecord rec;
    rec.prompt_tokens = prompt_tokens;
    tcp::resolver resolver(ioc);
    tcp::socket socket(ioc);
    try {
        auto endpoints = co_await resolver.async_resolve(host, port,
                                                         net::use_awaitable);
        co_await net::async_connect(socket, endpoints, net::use_awaitable);
    } catch (...) {
        co_return rec;  // ok stays false
    }

    std::string body = build_body(model, prompt, max_tokens);
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n";
    if (!api_key.empty())
        req << "Authorization: Bearer " << api_key << "\r\n";
    req << "Connection: close\r\n\r\n" << body;

    auto t0 = clock_type::now();
    auto first_token_at = t0;
    bool got_first = false;
    try {
        co_await net::async_write(socket, net::buffer(req.str()),
                                  net::use_awaitable);
        std::string pending;
        std::array<char, 8192> buf;
        for (;;) {
            std::size_t n = co_await socket.async_read_some(
                net::buffer(buf), net::use_awaitable);
            if (n == 0) break;
            pending.append(buf.data(), n);
            // Process every complete line; keep the partial tail.
            std::size_t pos;
            while ((pos = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, pos);
                pending.erase(0, pos + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.rfind("data:", 0) != 0) continue;
                std::string payload = line.substr(5);
                if (!payload.empty() && payload.front() == ' ')
                    payload.erase(0, 1);
                if (has_content(payload)) {
                    auto now = clock_type::now();
                    if (!got_first) {
                        got_first = true;
                        first_token_at = now;
                        rec.ttft_ms = ms_between(t0, now);
                    }
                    ++rec.output_tokens;
                }
            }
        }
    } catch (const boost::system::system_error& e) {
        // EOF arrives as an error on some stacks; a stream that delivered
        // tokens before closing is still a completed request.
        if (!got_first) co_return rec;
    } catch (...) {
        co_return rec;
    }

    auto done = clock_type::now();
    rec.latency_ms = ms_between(t0, done);
    if (rec.output_tokens > 1)
        rec.tpot_ms = ms_between(first_token_at, done) / (rec.output_tokens - 1);
    rec.ok = got_first && rec.output_tokens > 0;
    co_return rec;
}
