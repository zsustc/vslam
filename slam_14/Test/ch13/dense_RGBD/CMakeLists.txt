cmake_minimum_required(VERSION 3.10)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)

#project(dense_RGBD)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#eigen
include_directories("/usr/include/eigen3/")

#pcl
#set(PCL_DIR "/home/nuc/Downloads/pcl-pcl-1.8.0/build/PCLConfig.cmake")
#find_package(PCL REQUIRED COMPONENT common io filters)
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

#ocotomap
find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

add_executable(pointcloud_mapping pointcloud_mapping.cpp)
target_link_libraries(pointcloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(octo_mapping octo_mapping.cpp)
target_link_libraries(octo_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})
