cmake_minimum_required(VERSION 3.1)
project(g2o_curve_fitting)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

set(OpenCV_DIR /home/nuc/libs/opencv3.1/release/installed/share/OpenCV)
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_DIRS})

#set(G2O_INCLUDE_DIR /home/nuc/libs/g2o-20170730_git/installed/include/)
#message(Path:${G2O_INCLUDE_DIRS})
#set(G2O_LIB_DIRS /home/nuc/libs/g2o-20170730_git/installed/lib/)
#message(Path:${G2O_LIB_DIRS})

LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})
message(Path:${G2O_LIBRARY})
message(Path:${G2O_INCLUDE_DIR})

include_directories(/usr/include/eigen3)

add_executable(g2o_curve_fitting main.cpp)

target_link_libraries(g2o_curve_fitting ${OpenCV_LIBS} g2o_core g2o_stuff)


add_executable(Virtualtest test.cpp)
