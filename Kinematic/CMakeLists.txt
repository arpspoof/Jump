cmake_minimum_required(VERSION 3.11)

cmake_policy(SET CMP0078 NEW)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS"-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(EIGEN_DIR "" CACHE PATH "Source directory to eigen")
if (EIGEN_DIR STREQUAL "")
    message(FATAL_ERROR "No Eigen directory specified")
endif()
include_directories(SYSTEM "${EIGEN_DIR}")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(utils)
add_subdirectory(convhull)

project(KinematicCore)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

if(WIN32)
set(PYTHON_INCLUDE_PATH "" CACHE PATH "Python include dir, usually C:/Users/xx/AppData/Local/Programs/Python/Python37/include")
set(PYTHON_LIB "" CACHE PATH "Python lib dir, usually C:/Users/xx/AppData/Local/Programs/Python/Python37/libs")
if (PYTHON_INCLUDE_PATH STREQUAL "")
    message(FATAL_ERROR "No python include path specified")
endif()
if (PYTHON_LIB STREQUAL "")
    message(FATAL_ERROR "No python libs path specified")
endif()
else()
find_package(PythonLibs 3.7.4 REQUIRED)
find_package(PythonInterp ${PYTHONLIBS_VERSION_STRING} REQUIRED)
endif()

include_directories(${PYTHON_INCLUDE_PATH})
link_directories(${PYTHON_LIB})

set(CMAKE_SWIG_FLAGS "")

set_property(SOURCE KinematicCore.i PROPERTY CPLUSPLUS ON)
swig_add_library(${PROJECT_NAME} LANGUAGE python SOURCES KinematicCore.i KinematicCore.cpp)
swig_link_libraries(${PROJECT_NAME} utils convhull)

if(WIN32)
string(REPLACE / \\ CMAKE_BINARY_DIR_RV "${CMAKE_BINARY_DIR}")
string(REPLACE / \\ PROJECT_SOURCE_DIR_RV "${PROJECT_SOURCE_DIR}")
add_custom_target(copy_kinematic ALL
  COMMAND (if exist "${PROJECT_SOURCE_DIR_RV}\\py" rd /s /q "${PROJECT_SOURCE_DIR_RV}\\py")
  COMMAND mkdir "${PROJECT_SOURCE_DIR_RV}\\py"
  COMMAND xcopy /Y "${CMAKE_BINARY_DIR_RV}\\_KinematicCore.pyd" "${PROJECT_SOURCE_DIR_RV}\\py"
  COMMAND xcopy /Y "${CMAKE_BINARY_DIR_RV}\\KinematicCore.py" "${PROJECT_SOURCE_DIR_RV}\\py"
)
else()
add_custom_target(copy_kinematic ALL
  COMMAND mkdir -p "${PROJECT_SOURCE_DIR}/py"
  COMMAND cp -f "${CMAKE_BINARY_DIR}/_KinematicCore.so" "${CMAKE_BINARY_DIR}/KinematicCore.py" "${PROJECT_SOURCE_DIR}/py"
)
endif()

add_dependencies(copy_kinematic ${PROJECT_NAME})
