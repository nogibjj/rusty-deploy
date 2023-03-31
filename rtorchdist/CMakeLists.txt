cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(check_pytorch CXX)

find_package(Torch REQUIRED)

add_executable(check_pytorch check_pytorch.cpp)
target_link_libraries(check_pytorch "${TORCH_LIBRARIES}")
set_property(TARGET check_pytorch PROPERTY CXX_STANDARD 14)