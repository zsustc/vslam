cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_CXX_STANDARD 11)

#set(OpenCV_DIR "/home/nuc/libs/opencv3.1/release/installed")
set(OpenCV_DIR /home/nuc/libs/opencv3.1/release/installed/share/OpenCV)

LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(G2O REQUIRED)
find_package( OpenCV 3.1 REQUIRED )
find_package( CSparse REQUIRED )

include_directories( ${G2O_INCLUDE_DIR} )

find_package( CSparse REQUIRED )
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        "/usr/include/eigen3"
)


#if(OpenCV_FOUND)
#    include_directories(${OpenCV_INCLUDE_DIRS})
#    message(STATUS "OpenCV library status:")
#    message(STATUS "    version: ${OpenCV_VERSION}")
#    message(STATUS "    libraries: ${OpenCV_LIBS}")
#    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#else(OpenCV_FOUND)
#    message(FATAL_ERROR "OpenCV library not found")
#endif(OpenCV_FOUND)

add_executable( feature_extraction feature_extraction.cpp pose_estimation_3d3d.cpp)
target_link_libraries( feature_extraction ${OpenCV_LIBS} )

add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
target_link_libraries( pose_estimation_3d2d ${OpenCV_LIBS}  ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
