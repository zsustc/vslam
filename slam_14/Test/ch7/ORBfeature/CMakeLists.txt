cmake_minimum_required(VERSION 2.8)
project(ORBfeature)

set(CMAKE_CXX_STANDARD 11)

#set(OpenCV_DIR "/home/nuc/libs/opencv3.1/release/installed")
set(OpenCV_DIR /home/nuc/libs/opencv3.1/release/installed/share/OpenCV)

find_package( OpenCV 3.1 REQUIRED )

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else(OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
endif(OpenCV_FOUND)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(ORBfeature main.cpp)
target_link_libraries( ORBfeature ${OpenCV_LIBS} )