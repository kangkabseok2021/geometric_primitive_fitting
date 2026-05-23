#include <gtest/gtest.h>
#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/eval.h>
#include "PythonEngine.h"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <numbers>

namespace nb = nanobind;
namespace fs = std::filesystem;

// SCRIPTS_DIR is injected by CMake as a compile definition.
#ifndef SCRIPTS_DIR
#define SCRIPTS_DIR "solver/scripts"
#endif

static const fs::path kScriptsDir{SCRIPTS_DIR};
static const fs::path kUserRhs = kScriptsDir / "user_rhs.py";

// --- Tests ---

TEST(PythonEngine, LoadScriptOk) {
    PythonEngine eng;
    EXPECT_NO_THROW(eng.loadScript(kUserRhs));
}

TEST(PythonEngine, CallFunctionReturnsCorrectValue) {
    PythonEngine eng;
    eng.loadScript(kUserRhs);
    // rhs(y=1.0, t=0.0) = -0.5*1.0 + 0.1*0 = -0.5
    double result = eng.callFunction("rhs", 1.0, 0.0);
    EXPECT_NEAR(result, -0.5, 1e-12);
}

TEST(PythonEngine, CallUnknownFunctionThrows) {
    PythonEngine eng;
    eng.loadScript(kUserRhs);
    EXPECT_THROW(eng.callFunction("nonexistent_fn", 1.0, 0.0), std::runtime_error);
}

TEST(PythonEngine, MalformedScriptThrowsWithTraceback) {
    // Write a syntactically broken script to a temp file.
    fs::path bad = fs::temp_directory_path() / "bad_script_test.py";
    { std::ofstream f(bad); f << "def broken(:\n    pass\n"; }
    PythonEngine eng;
    EXPECT_THROW(eng.loadScript(bad), std::runtime_error);
    fs::remove(bad);
}

TEST(PythonEngine, StiffRhsCallable) {
    PythonEngine eng;
    eng.loadScript(kUserRhs);
    // rhs_stiff(y=0.0, t=1.0) = -1000*(0 - 1) + 2 = 1002
    double result = eng.callFunction("rhs_stiff", 0.0, 1.0);
    EXPECT_NEAR(result, 1002.0, 1e-9);
}

TEST(PythonEngine, ScopedInterpreterRaii) {
    // Interpreter is already running (managed by main); creating an engine is safe.
    EXPECT_NO_THROW({
        PythonEngine eng;
        eng.loadScript(kUserRhs);
        eng.callFunction("rhs", 2.0, 1.0);
    });
}

TEST(PythonEngine, FloatPrecisionPreserved) {
    PythonEngine eng;
    eng.loadScript(kUserRhs);
    // rhs(y, t=0) = -0.5*y → for y=2.718281828, result = -1.359140914
    const double y_val = std::numbers::e;
    double result = eng.callFunction("rhs", y_val, 0.0);
    EXPECT_NEAR(result, -0.5 * y_val, 1e-14);
}

TEST(PythonEngine, NoneReturnThrows) {
    PythonEngine eng;
    // Define an inline function that returns None
    nb::object g = nb::module_::import_("__main__").attr("__dict__");
    nb::exec("def returns_none(y, t): return None", g);
    EXPECT_THROW(eng.callFunction("returns_none", 1.0, 0.0), std::runtime_error);
}

TEST(PythonEngine, CallVectorFunction) {
    PythonEngine eng;
    eng.loadScript(kUserRhs);
    // batch_rhs(t_array) = exp(-t_array)
    std::vector<double> t_grid = {0.0, 1.0, 2.0};
    auto result = eng.callVectorFunction("batch_rhs", t_grid);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0], 1.0,               1e-12);
    EXPECT_NEAR(result[1], std::exp(-1.0),    1e-12);
    EXPECT_NEAR(result[2], std::exp(-2.0),    1e-12);
}

// Custom main: initialise the CPython interpreter once for the entire test binary.
int main(int argc, char** argv) {
    Py_Initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();
    Py_Finalize();
    return rc;
}
