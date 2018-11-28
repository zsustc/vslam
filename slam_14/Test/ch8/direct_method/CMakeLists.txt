cmake_minimum_required(VERSION 3.10)
project(direct_method)

set(CMAKE_CXX_STANDARD 11)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(G2O REQUIRED)
include_directories(&{G2O_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_LIBS})

include_directories("/usr/include/eigen3")
set(G2O_LIBS
g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension
)


add_executable(direct_method direct_sparse.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${G2O_LIBS} ${G2O_LIBS})