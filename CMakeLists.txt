cmake_minimum_required(VERSION 3.20)
project(geometric_primitive_fitting CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Eigen — header-only: populate without running its CMakeLists.txt
# (avoids registering Eigen's 900+ internal tests in ctest)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
add_library(Eigen3 INTERFACE)
add_library(Eigen3::Eigen ALIAS Eigen3)
target_include_directories(Eigen3 INTERFACE ${eigen_SOURCE_DIR})

# GoogleTest
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# nanobind (replaces pybind11)
# GIT_SUBMODULES "ext/robin_map": explicitly initialise only the robin_map submodule
# that nanobind requires; avoids fetching all submodules or needing a full recursive clone.
FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.4.0
  GIT_SUBMODULES "ext/robin_map"
)
# Pre-find Python in the parent scope so Python_INCLUDE_DIRS is set when
# nanobind_build_library() is called — FetchContent sub-scopes don't propagate variables.
find_package(Python 3.8 COMPONENTS Interpreter Development.Module QUIET)
FetchContent_MakeAvailable(nanobind)

# --- Sphere fitter library ---
add_library(sphere_fitter_lib
  src/PointCloud.cpp
  src/SphereFitter.cpp
  src/RansacFilter.cpp
)
target_include_directories(sphere_fitter_lib PUBLIC include)
target_link_libraries(sphere_fitter_lib PUBLIC Eigen3::Eigen)
set_target_properties(sphere_fitter_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(sphere_fitter_lib PRIVATE -Wall -Wextra -O2)

# --- Solver core (header-only: Concepts, MathCore, Complex, SolverBase, Solvers, SimulationViews) ---
add_library(solver_core INTERFACE)
target_include_directories(solver_core INTERFACE solver/include)
target_compile_features(solver_core INTERFACE cxx_std_23)

# --- Python targets (only when Python3 development headers are available) ---
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
  # nanobind sphere-fitter extension module — also creates the `nanobind` SHARED target
  nanobind_add_module(sphere_fitter_py python/bindings.cpp)
  target_link_libraries(sphere_fitter_py PRIVATE sphere_fitter_lib)

  # nanobind solver extension module (EulerSolver + RK4Solver callable from Python)
  # NB_STATIC: builds nanobind-static alongside the SHARED nanobind, needed by python_engine_lib
  nanobind_add_module(solver_py NB_STATIC solver/python/solver_bindings.cpp)
  target_link_libraries(solver_py PRIVATE solver_core)

  # PythonEngine — embeds CPython via nanobind for runtime-defined RHS functions.
  # Links nanobind-static so the embedding binary is self-contained (no RPATH dependency).
  add_library(python_engine_lib STATIC solver/src/PythonEngine.cpp)
  target_include_directories(python_engine_lib PUBLIC solver/include)
  target_link_libraries(python_engine_lib
    PUBLIC  solver_core
    PRIVATE nanobind-static Python3::Python)
  set_target_properties(python_engine_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
  
  # Skeleton kinematics optimizer C++ acceleration
  nanobind_add_module(skeleton_cpp skeleton_cpp/bindings.cpp skeleton_cpp/lm_solver.cpp)
  target_include_directories(skeleton_cpp PRIVATE skeleton_cpp)
  target_link_libraries(skeleton_cpp PRIVATE Eigen3::Eigen)
  target_compile_options(skeleton_cpp PRIVATE -O3 -march=native -ffast-math)
  set_target_properties(skeleton_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/skeleton_optimizer)
endif()

enable_testing()
add_subdirectory(tests)
