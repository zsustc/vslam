cmake_minimum_required(VERSION 3.10)
project(myslam)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_BUILD_TYPE "Release")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


############################## dependencies ############################################
# Eigen
include_directories("/usr/include/eigen3")

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

#G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})


set(THIRD_PARTY_LIBS
        ${G2O_LIBS}
        ${PROJECT_SOURCE_DIR}
        ${Sophus_LIBRARIES}
        g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension
        )

add_executable(0_1 main.cpp)
target_link_libraries(0_1 ${OpenCV_LIBS} ${G2O_LIBS} ${Sophus_LIBRARIES})

#################################### source and test ######################################
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(test)