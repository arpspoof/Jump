cmake_minimum_required(VERSION 3.11)

project(convhull)

file(GLOB source_files *.cpp)
add_library(${PROJECT_NAME} ${source_files})
