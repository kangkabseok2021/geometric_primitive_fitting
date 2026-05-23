#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include "Solvers.h"

namespace nb = nanobind;
using namespace solver;

// Helper: wrap CRTP solve() for Python — accepts a Python callable as the RHS.
template<typename S>
static std::vector<double> py_solve(const S& s,
                                     std::function<double(double, double)> f,
                                     double t0, double t1, double y0, double dt) {
    return s.solve(f, t0, t1, y0, dt);
}

NB_MODULE(solver_py, m) {
    m.doc() = "C++23 CRTP ODE solvers — zero-vtable Euler and RK4";

    nb::class_<EulerSolver>(m, "EulerSolver")
        .def(nb::init<>())
        .def("solve", &py_solve<EulerSolver>,
             nb::arg("f"), nb::arg("t0"), nb::arg("t1"),
             nb::arg("y0"), nb::arg("dt"),
             "Integrate dy/dt=f(t,y) with Euler method; returns list of y values.");

    nb::class_<RK4Solver>(m, "RK4Solver")
        .def(nb::init<>())
        .def("solve", &py_solve<RK4Solver>,
             nb::arg("f"), nb::arg("t0"), nb::arg("t1"),
             nb::arg("y0"), nb::arg("dt"),
             "Integrate dy/dt=f(t,y) with RK4; returns list of y values.");
}
