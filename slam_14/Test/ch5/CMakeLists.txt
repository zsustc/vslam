cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Debug")

project(imageBasics)

set(OpenCV_DIR /home/nuc/libs/opencv-trunk/release/installed/share/OpenCV)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLDE_DIRS})

add_executable(imageBasics imageBasics.cpp)

target_link_libraries(imageBasics ${OpenCV_LIBS})

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()
