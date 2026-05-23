#pragma once
#include <filesystem>
#include <string_view>
#include <vector>

// Embedded Python interpreter wrapper using nanobind.
// The interpreter lifetime is managed externally (nb::scoped_interpreter in main or test fixture).
// Each PythonEngine instance holds a reference to the __main__ module for function lookup.
class PythonEngine {
public:
    // Execute a Python script file; functions it defines become callable.
    void loadScript(const std::filesystem::path& script_path);

    // Call a Python function f(y, t) → double (scalar ODE right-hand side).
    [[nodiscard]] double callFunction(std::string_view func_name, double y, double t) const;

    // Call a Python function f(t_array) → numpy array (vectorised RHS over a time grid).
    // Returns the result as std::vector<double>. Zero-copy for the input array.
    [[nodiscard]] std::vector<double> callVectorFunction(
        std::string_view func_name, const std::vector<double>& t_grid) const;
};
