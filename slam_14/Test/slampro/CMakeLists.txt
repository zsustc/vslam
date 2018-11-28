cmake_minimum_required(VERSION 2.6)
project(slampro)

add_executable(slampro main.cpp)

install(TARGETS slampro RUNTIME DESTINATION bin)
