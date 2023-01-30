cmake_minimum_required(VERSION 3.24)
project(test1 CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

include_directories(.)
add_subdirectory(lib/glfw)
add_subdirectory(lib/FFmpeg)
add_subdirectory(lib/matplotplusplus)

add_executable(test1 src/cud/functs.cu src/entry.cpp src/cud/functs.h src/vidStuff/vidReader.h src/vidStuff/vidReader.cpp src/helper/helperFuncts.cpp src/helper/helperFuncts.h)

target_link_libraries(test1 FFmpeg glfw matplot)

set_target_properties(test1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
