# Minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name
project(RayTracer)

# Set the C++ compiler and standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable target
add_executable(rt main.cc random_utils.cpp)

# Optimization flag
target_compile_options(rt PRIVATE -O3)

# Custom command for generating out.ppm
add_custom_target(run ALL
    COMMAND ./rt > out.ppm
    DEPENDS rt
)

# Clean files
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove -f rt out.ppm out.jpg
)
