cmake_minimum_required(VERSION 3.24)
project(test1 CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

include_directories(.)
add_subdirectory(lib/glfw)
add_subdirectory(lib/FFmpeg)
add_subdirectory(lib/matplotplusplus)

add_definitions(-DGL_SILENCE_DEPRECATION)

list(APPEND SOURCES
        src/entry.cpp
        src/vidStuff/vidReader.cpp
        src/vidStuff/vidReader.h
        src/cud/functs.cu
        src/cud/functs.h
        )

add_executable(test1 src/cud/functs.cu src/entry.cpp src/cud/functs.h src/vidStuff/vidReader.h src/vidStuff/vidReader.cpp src/helper/helperFuncts.cpp src/helper/helperFuncts.h ${SOURCES} src/neurons/Neuron.cpp src/neurons/Neuron.h src/cud/rgc.cu src/cud/rgc.cuh src/cud/mainseq.cpp src/cud/mainseq.h)



if(APPLE)
    list(APPEND EXTRA_LIBS
            "-framework OpenGL"
            )

    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/MacOSXBundleInfo.plist.in
            ${CMAKE_CURRENT_BINARY_DIR}/assets/MacOSXBundleInfo.plist
    )

    set_target_properties(video-app PROPERTIES
            MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_BINARY_DIR}/assets/MacOSXBundleInfo.plist
            )

elseif(WIN32)
    list(APPEND EXTRA_LIBS
            "-lglu32 -lopengl32"
            )
    set(CMAKE_EXE_LINKER_FLAGS "-std=gnu99 -static -static-libgcc -static-libstdc++ -mwindows")
else()
    list(APPEND EXTRA_LIBS
            "-lGL -lGLU -lX11"
            )
endif()

target_link_libraries(test1 FFmpeg glfw ${EXTRA_LIBS} matplot)

set_target_properties(test1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
