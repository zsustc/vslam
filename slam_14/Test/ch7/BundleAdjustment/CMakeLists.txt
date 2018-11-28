cmake_minimum_required(VERSION 3.10)
project(Vo)

#set(CMAKE_CXX_STANDARD 11)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#g2o
find_package(G2O REQUIRED)

#opencv
find_package(OpenCV 3.1 REQUIRED)

#csparse
find_package(CSparse REQUIRED)

#eigen
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        "/usr/include/eigen3/"
)

#add_executable(BundleAdjustment pose_estimation_3d2d.cpp )
#target_link_libraries(BundleAdjustment ${OpenCV_LIBS} ${CSPARSE_LIBRARY}
#        g2o_core g2o_types_sba g2o_stuff g2o_csparse_extension)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d
        ${OpenCV_LIBS}
        g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
        ${CSPARSE_LIBRARY}
        )