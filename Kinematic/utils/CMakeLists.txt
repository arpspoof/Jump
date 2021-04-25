cmake_minimum_required(VERSION 3.11)

project(utils)

set(COMPILE_STATUS checked)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB source_files *.cpp)
add_library(${PROJECT_NAME} ${source_files})
