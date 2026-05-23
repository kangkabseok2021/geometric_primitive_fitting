add_executable(sphere_tests test_sphere_fitter.cpp test_ransac.cpp)
target_link_libraries(sphere_tests PRIVATE sphere_fitter_lib GTest::gtest_main)
target_include_directories(sphere_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(sphere_tests)
