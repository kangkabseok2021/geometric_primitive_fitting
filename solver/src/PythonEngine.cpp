#include "PythonEngine.h"
#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/eval.h>
#include <nanobind/stl/vector.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

void PythonEngine::loadScript(const std::filesystem::path& script_path) {
    std::ifstream file(script_path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open script: " + script_path.string());
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string code = ss.str();
    try {
        nb::object main_globals = nb::module_::import_("__main__").attr("__dict__");
        nb::exec(nb::str{code.c_str()}, main_globals);
    } catch (const nb::python_error& e) {
        throw std::runtime_error(std::string("Failed to load script: ") + e.what());
    }
}

double PythonEngine::callFunction(std::string_view func_name, double y, double t) const {
    try {
        nb::object fn = nb::module_::import_("__main__").attr(func_name.data());
        return nb::cast<double>(fn(y, t));
    } catch (const nb::python_error& e) {
        throw std::runtime_error(std::string("Python call '") +
                                 std::string(func_name) + "' failed: " + e.what());
    } catch (const nb::cast_error&) {
        throw std::runtime_error(std::string("'") + std::string(func_name) +
                                 "' did not return a float");
    }
}

std::vector<double> PythonEngine::callVectorFunction(std::string_view func_name,
                                                      const std::vector<double>& t_grid) const {
    try {
        // Pass the grid as a Python list (safe in embedded context; avoids numpy C API init).
        nb::list py_list;
        for (double v : t_grid)
            py_list.append(v);

        nb::object fn = nb::module_::import_("__main__").attr(func_name.data());
        nb::object result = fn(py_list);

        // Convert result (numpy array or sequence) back to std::vector<double>.
        std::vector<double> out;
        for (auto item : nb::cast<nb::sequence>(result))
            out.push_back(nb::cast<double>(item));
        return out;
    } catch (const nb::python_error& e) {
        throw std::runtime_error(std::string("Python vector call '") +
                                 std::string(func_name) + "' failed: " + e.what());
    }
}
