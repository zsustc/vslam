cmake_minimum_required(VERSION 2.8) 

set(CMAKE_BUILD_TYPE "Debug")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set( CMAKE_CXX_FLAGS "-std=c++11" )

project(joinMap)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#eigen
include_directories("/usr/include/eigen3")

#pcl
set(PCL_DIR "/home/nuc/Downloads/pcl-pcl-1.8.0/build/PCLConfig.cmake")

find_package(PCL REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable(joinMap joinMap.cpp)

target_link_libraries(joinMap ${OpenCV_LIBS} ${PCL_LIBRARIES})

