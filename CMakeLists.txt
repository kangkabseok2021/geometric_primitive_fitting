cmake_minimum_required(VERSION 3.20)
project(geometric_primitive_fitting CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Eigen
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

# GoogleTest
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# pybind11
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

# Core library
add_library(sphere_fitter_lib
  src/PointCloud.cpp
  src/SphereFitter.cpp
  src/RansacFilter.cpp
)
target_include_directories(sphere_fitter_lib PUBLIC include)
target_link_libraries(sphere_fitter_lib PUBLIC Eigen3::Eigen)
target_compile_options(sphere_fitter_lib PRIVATE -Wall -Wextra -O2)

# pybind11 module
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
  pybind11_add_module(sphere_fitter_py python/bindings.cpp)
  target_link_libraries(sphere_fitter_py PRIVATE sphere_fitter_lib)
endif()

# Tests
enable_testing()
add_subdirectory(tests)
