#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "SphereFitter.h"
#include "RansacFilter.h"
#include "PointCloud.h"

namespace nb = nanobind;

NB_MODULE(sphere_fitter_py, m) {
    m.doc() = "Geometric Primitive Fitting — Levenberg-Marquardt sphere fitter with RANSAC";

    nb::class_<SphereParams>(m, "SphereParams")
        .def_ro("center",       &SphereParams::center)
        .def_ro("radius",       &SphereParams::radius)
        .def_ro("rms_residual", &SphereParams::rms_residual)
        .def_ro("iterations",   &SphereParams::iterations)
        .def_ro("converged",    &SphereParams::converged)
        .def("__repr__", [](const SphereParams& p) {
            return "<SphereParams center=[" +
                   std::to_string(p.center[0]) + "," +
                   std::to_string(p.center[1]) + "," +
                   std::to_string(p.center[2]) + "] radius=" +
                   std::to_string(p.radius) + " rms=" +
                   std::to_string(p.rms_residual) + ">";
        });

    nb::class_<SphereFitterOptions>(m, "SphereFitterOptions")
        .def(nb::init<>())
        .def_rw("max_iter",    &SphereFitterOptions::max_iter)
        .def_rw("tol",         &SphereFitterOptions::tol)
        .def_rw("lambda_init", &SphereFitterOptions::lambda_init)
        .def_rw("lambda_max",  &SphereFitterOptions::lambda_max)
        .def_rw("lambda_min",  &SphereFitterOptions::lambda_min);

    nb::class_<SphereFitter>(m, "SphereFitter")
        .def(nb::init<SphereFitterOptions>(), nb::arg("opts") = SphereFitterOptions{})
        .def("fit",                &SphereFitter::fit)
        .def_static("residual",     &SphereFitter::residual)
        .def_static("jacobian_row", &SphereFitter::jacobian_row);

    nb::class_<RansacResult>(m, "RansacResult")
        .def_ro("center",  &RansacResult::center)
        .def_ro("radius",  &RansacResult::radius)
        .def_ro("inliers", &RansacResult::inliers)
        .def_ro("success", &RansacResult::success);

    m.def("generate_noisy_sphere", &generate_noisy_sphere,
          nb::arg("center"), nb::arg("radius"), nb::arg("n_points"),
          nb::arg("sigma"), nb::arg("seed") = 42u);

    m.def("ransac_sphere", &ransac_sphere,
          nb::arg("pts"),
          nb::arg("inlier_threshold") = 0.05,
          nb::arg("max_iter")         = 1000,
          nb::arg("seed")             = 42u,
          nb::arg("opts")             = SphereFitterOptions{});
}
