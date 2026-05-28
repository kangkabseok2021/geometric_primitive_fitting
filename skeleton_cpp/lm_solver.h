#pragma once

#include <Eigen/Dense>

namespace skeleton {

struct LMResult {
    Eigen::Vector3d theta;
    double residual;
    int iterations;
    bool converged;
};

// Forward kinematics using Rodrigues rotation formula
Eigen::Matrix3d forward_kinematics(const Eigen::Vector3d& theta, double L1, double L2);

// Analytic Jacobian for the forward kinematics
Eigen::Matrix<double, 9, 3> analytic_jacobian(const Eigen::Vector3d& theta, double L1, double L2);

// Solve for a single frame
LMResult solve_lm(const Eigen::Matrix3d& observed, 
                  const Eigen::Vector3d& weights, 
                  double L1, 
                  double L2, 
                  const Eigen::Vector3d& theta_init, 
                  int max_iter = 100, 
                  double tol = 1e-10);

} // namespace skeleton
