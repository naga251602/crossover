cmake_minimum_required(VERSION 3.20)
project(serving_harness CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options json)
find_package(Threads REQUIRED)

add_executable(harness
    src/main.cpp
    src/config.cpp
    src/workload.cpp
    src/client.cpp
)
target_link_libraries(harness PRIVATE
    Boost::program_options
    Boost::json
    Threads::Threads
)

enable_testing()
