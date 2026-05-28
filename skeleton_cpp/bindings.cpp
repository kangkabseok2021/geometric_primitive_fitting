#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include "lm_solver.h"
#include <vector>

namespace nb = nanobind;

using namespace skeleton;

nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> solve_sequence_cpp(
        nb::ndarray<nb::numpy, double, nb::shape<-1, 3, 3>> observed_seq,
        nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> weights_seq,
        double L1,
        double L2,
        Eigen::Vector3d theta_init,
        int max_iter) {

    size_t N = observed_seq.shape(0);
    
    // Allocate output array
    double* out_data = new double[N * 3];
    nb::capsule owner(out_data, [](void *p) noexcept { delete[] (double *) p; });
    size_t shape[2] = { N, 3 };
    
    auto result = nb::ndarray<nb::numpy, double, nb::shape<-1, 3>>(
        out_data, 2, shape, owner
    );

    auto obs_view = observed_seq.view();
    auto w_view = weights_seq.view();
    
    for (size_t i = 0; i < N; ++i) {
        Eigen::Matrix3d obs;
        obs << obs_view(i, 0, 0), obs_view(i, 0, 1), obs_view(i, 0, 2),
               obs_view(i, 1, 0), obs_view(i, 1, 1), obs_view(i, 1, 2),
               obs_view(i, 2, 0), obs_view(i, 2, 1), obs_view(i, 2, 2);
               
        Eigen::Vector3d w(w_view(i, 0), w_view(i, 1), w_view(i, 2));
        
        LMResult r = solve_lm(obs, w, L1, L2, theta_init, max_iter, 1e-10);
        theta_init = r.theta; // warm start
        
        out_data[i * 3 + 0] = r.theta(0);
        out_data[i * 3 + 1] = r.theta(1);
        out_data[i * 3 + 2] = r.theta(2);
    }
    
    return result;
}

NB_MODULE(skeleton_cpp, m) {
    m.doc() = "C++ LM Solver for Skeletal Kinematics";
    
    nb::class_<LMResult>(m, "LMResult")
        .def_ro("theta", &LMResult::theta)
        .def_ro("residual", &LMResult::residual)
        .def_ro("iterations", &LMResult::iterations)
        .def_ro("converged", &LMResult::converged);
        
    m.def("solve_lm", &solve_lm, 
          nb::arg("observed"), nb::arg("weights"), nb::arg("L1"), nb::arg("L2"), 
          nb::arg("theta_init"), nb::arg("max_iter")=100, nb::arg("tol")=1e-10,
          "Solve LM for a single frame");
          
    m.def("solve_sequence", &solve_sequence_cpp,
          nb::arg("observed_seq"), nb::arg("weights_seq"), nb::arg("L1"), nb::arg("L2"),
          nb::arg("theta_init"), nb::arg("max_iter")=100,
          "Solve LM for an entire sequence");
}
