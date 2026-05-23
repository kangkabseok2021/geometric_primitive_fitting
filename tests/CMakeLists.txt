include(GoogleTest)

# Sphere fitter tests (existing: 10 GoogleTests)
add_executable(sphere_tests test_sphere_fitter.cpp test_ransac.cpp)
target_link_libraries(sphere_tests PRIVATE sphere_fitter_lib GTest::gtest_main)
target_include_directories(sphere_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
gtest_discover_tests(sphere_tests)

# Solver tests: MathCore compile-time proofs + ODE accuracy (new)
add_executable(solver_tests test_math_core.cpp test_solvers.cpp)
target_link_libraries(solver_tests PRIVATE solver_core GTest::gtest_main)
gtest_discover_tests(solver_tests)

# PythonEngine tests — only built when python_engine_lib is available
if(TARGET python_engine_lib)
  add_executable(python_engine_tests test_python_engine.cpp)
  target_link_libraries(python_engine_tests PRIVATE
    python_engine_lib GTest::gtest nanobind-static Python3::Python)
  target_compile_definitions(python_engine_tests PRIVATE
    SCRIPTS_DIR="${CMAKE_SOURCE_DIR}/solver/scripts")
  gtest_discover_tests(python_engine_tests)
endif()
