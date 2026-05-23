#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "SphereFitter.h"
#include "RansacFilter.h"
#include "PointCloud.h"

namespace py = pybind11;

PYBIND11_MODULE(sphere_fitter_py, m) {
    m.doc() = "Geometric Primitive Fitting — Levenberg-Marquardt sphere fitter with RANSAC";

    py::class_<SphereParams>(m, "SphereParams")
        .def_readonly("center",       &SphereParams::center)
        .def_readonly("radius",       &SphereParams::radius)
        .def_readonly("rms_residual", &SphereParams::rms_residual)
        .def_readonly("iterations",   &SphereParams::iterations)
        .def_readonly("converged",    &SphereParams::converged)
        .def("__repr__", [](const SphereParams& p) {
            return "<SphereParams center=[" +
                   std::to_string(p.center[0]) + "," +
                   std::to_string(p.center[1]) + "," +
                   std::to_string(p.center[2]) + "] radius=" +
                   std::to_string(p.radius) + " rms=" +
                   std::to_string(p.rms_residual) + ">";
        });

    py::class_<SphereFitterOptions>(m, "SphereFitterOptions")
        .def(py::init<>())
        .def_readwrite("max_iter",    &SphereFitterOptions::max_iter)
        .def_readwrite("tol",         &SphereFitterOptions::tol)
        .def_readwrite("lambda_init", &SphereFitterOptions::lambda_init)
        .def_readwrite("lambda_max",  &SphereFitterOptions::lambda_max)
        .def_readwrite("lambda_min",  &SphereFitterOptions::lambda_min);

    py::class_<SphereFitter>(m, "SphereFitter")
        .def(py::init<SphereFitterOptions>(), py::arg("opts") = SphereFitterOptions{})
        .def("fit",          &SphereFitter::fit)
        .def_static("residual",     &SphereFitter::residual)
        .def_static("jacobian_row", &SphereFitter::jacobian_row);

    py::class_<RansacResult>(m, "RansacResult")
        .def_readonly("center",  &RansacResult::center)
        .def_readonly("radius",  &RansacResult::radius)
        .def_readonly("inliers", &RansacResult::inliers)
        .def_readonly("success", &RansacResult::success);

    m.def("generate_noisy_sphere", &generate_noisy_sphere,
          py::arg("center"), py::arg("radius"), py::arg("n_points"),
          py::arg("sigma"), py::arg("seed") = 42u);

    m.def("ransac_sphere", &ransac_sphere,
          py::arg("pts"),
          py::arg("inlier_threshold") = 0.05,
          py::arg("max_iter")         = 1000,
          py::arg("seed")             = 42u,
          py::arg("opts")             = SphereFitterOptions{});
}
