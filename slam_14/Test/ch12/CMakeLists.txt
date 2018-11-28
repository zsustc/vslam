cmake_minimum_required(VERSION 3.10)
project(ch12)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
#set(CMAKE_CXX_STANDARD 11)

#opencv
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Eigen
include_directories("/usr/include/eigen3")

#dbow3
find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_INCLUDE_DIRS})

#set(DBoW3_LIBS "/usr/local/lib/libDBoW3.a")


add_executable(feature_training feature_training.cpp)
target_link_libraries(feature_training ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(loop_closure loop_closure.cpp)
target_link_libraries(loop_closure ${OpenCV_LIBS} ${DBoW3_LIBS})