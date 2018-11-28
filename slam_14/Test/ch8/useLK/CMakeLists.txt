cmake_minimum_required(VERSION 3.10)
project(useLK)

set(CMAKE_CXX_STANDARD 11)

#opencv
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_LIBS})

add_executable(useLK useLK.cpp)
target_link_libraries(useLK ${OpenCV_LIBS})