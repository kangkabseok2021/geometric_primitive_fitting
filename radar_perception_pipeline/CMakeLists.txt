cmake_minimum_required(VERSION 3.20)
project(radar_perception_pipeline CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ─────────────────────────────────────────────────────────────
include(FetchContent)

# Eigen (reuse parent's fetch if available, else fetch standalone)
if(NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG        3.4.0
      GIT_SHALLOW    TRUE
    )
    set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# nlohmann/json
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# GoogleTest (reuse parent's if available)
if(NOT TARGET GTest::gtest_main)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
      GIT_SHALLOW    TRUE
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ── Pipeline binary ──────────────────────────────────────────────────────────
add_executable(radar_pipeline src/main.cpp)
target_include_directories(radar_pipeline PRIVATE include)
target_link_libraries(radar_pipeline PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
target_compile_options(radar_pipeline PRIVATE -Wall -Wextra)

# ── Unit tests ───────────────────────────────────────────────────────────────
enable_testing()
add_executable(radar_tests tests/test_radar.cpp)
target_include_directories(radar_tests PRIVATE include)
target_link_libraries(radar_tests PRIVATE GTest::gtest_main Eigen3::Eigen)
target_compile_options(radar_tests PRIVATE -Wall -Wextra)

include(GoogleTest)
gtest_discover_tests(radar_tests DISCOVERY_MODE PRE_TEST)
