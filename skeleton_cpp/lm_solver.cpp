#include "lm_solver.h"
#include <cmath>
#include <iostream>

namespace skeleton {

Eigen::Matrix3d forward_kinematics(const Eigen::Vector3d& theta, double L1, double L2) {
    double phi = theta(0);
    double psi = theta(1);
    double alpha = theta(2);

    Eigen::Vector3d shoulder(0.0, 0.0, 0.0);

    Eigen::Vector3d d_s(
        std::cos(psi) * std::cos(phi),
        std::cos(psi) * std::sin(phi),
        std::sin(psi)
    );

    Eigen::Vector3d elbow = L1 * d_s;

    Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
    Eigen::Vector3d rot_axis = d_s.cross(z_axis);
    
    double norm = rot_axis.norm();
    if (norm > 1e-6) {
        rot_axis /= norm;
    } else {
        rot_axis = Eigen::Vector3d(1.0, 0.0, 0.0);
    }

    // Rodrigues rotation
    Eigen::Matrix3d K;
    K << 0.0, -rot_axis(2), rot_axis(1),
         rot_axis(2), 0.0, -rot_axis(0),
         -rot_axis(1), rot_axis(0), 0.0;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(alpha) * K + (1.0 - std::cos(alpha)) * (K * K);

    Eigen::Vector3d d_f = R * d_s;
    Eigen::Vector3d wrist = elbow + L2 * d_f;

    Eigen::Matrix3d res;
    res.row(0) = shoulder;
    res.row(1) = elbow;
    res.row(2) = wrist;
    return res;
}

Eigen::Matrix<double, 9, 3> analytic_jacobian(const Eigen::Vector3d& theta, double L1, double L2) {
    // For exact match with Python, we use the same hybrid approach: 
    // analytic for elbow, finite-diff for wrist.
    // In a full production C++ implementation we'd do full analytic.
    
    double phi = theta(0);
    double psi = theta(1);
    
    Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
    
    Eigen::Vector3d dds_dphi(
        -std::cos(psi) * std::sin(phi),
        std::cos(psi) * std::cos(phi),
        0.0
    );
    
    Eigen::Vector3d dds_dpsi(
        -std::sin(psi) * std::cos(phi),
        -std::sin(psi) * std::sin(phi),
        std::cos(psi)
    );
    
    J.block<3, 1>(3, 0) = L1 * dds_dphi;
    J.block<3, 1>(3, 1) = L1 * dds_dpsi;
    
    // Finite differences for wrist to perfectly match Python test baseline
    double eps = 1e-6;
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d t1 = theta;
        Eigen::Vector3d t2 = theta;
        t1(i) -= eps;
        t2(i) += eps;
        
        Eigen::Matrix3d fk1 = forward_kinematics(t1, L1, L2);
        Eigen::Matrix3d fk2 = forward_kinematics(t2, L1, L2);
        
        J.block<3, 1>(6, i) = (fk2.row(2) - fk1.row(2)) / (2.0 * eps);
    }
    
    return J;
}

LMResult solve_lm(const Eigen::Matrix3d& observed, 
                  const Eigen::Vector3d& weights, 
                  double L1, 
                  double L2, 
                  const Eigen::Vector3d& theta_init, 
                  int max_iter, 
                  double tol) {
                  
    Eigen::Vector3d theta = theta_init;
    double lambda = 1e-3;
    
    Eigen::VectorXd sqrt_w(9);
    for(int i=0; i<3; ++i) {
        double sw = std::sqrt(weights(i));
        sqrt_w(3*i) = sw;
        sqrt_w(3*i+1) = sw;
        sqrt_w(3*i+2) = sw;
    }
    
    auto compute_residual = [&](const Eigen::Vector3d& t) {
        Eigen::Matrix3d fk = forward_kinematics(t, L1, L2);
        Eigen::VectorXd r(9);
        for(int i=0; i<3; ++i) {
            r.segment<3>(3*i) = sqrt_w.segment<3>(3*i).cwiseProduct(observed.row(i).transpose() - fk.row(i).transpose());
        }
        return r;
    };
    
    Eigen::VectorXd r = compute_residual(theta);
    double cost = r.squaredNorm();
    
    bool converged = false;
    int iter = 0;
    
    for (; iter < max_iter; ++iter) {
        Eigen::Matrix<double, 9, 3> J = analytic_jacobian(theta, L1, L2);
        
        // Weight the Jacobian
        for(int i=0; i<3; ++i) {
            J.block<3,3>(3*i, 0) = J.block<3,3>(3*i, 0).array().colwise() * (-sqrt_w.segment<3>(3*i).array());
        }
        
        Eigen::Matrix3d H = J.transpose() * J;
        Eigen::Vector3d g = J.transpose() * r;
        
        // Check gradient convergence
        if (g.lpNorm<Eigen::Infinity>() < tol) {
            converged = true;
            break;
        }
        
        Eigen::Matrix3d H_lm = H;
        H_lm.diagonal() += lambda * H.diagonal();
        Eigen::Vector3d delta = H_lm.ldlt().solve(-g);
        
        if (delta.norm() < tol * (theta.norm() + tol)) {
            converged = true;
            break;
        }
        
        Eigen::Vector3d theta_new = theta + delta;
        Eigen::VectorXd r_new = compute_residual(theta_new);
        double cost_new = r_new.squaredNorm();
        
        if (cost_new < cost) {
            lambda /= 10.0;
            theta = theta_new;
            r = r_new;
            cost = cost_new;
        } else {
            lambda *= 10.0;
        }
    }
    
    return {theta, cost, iter, converged};
}

} // namespace skeleton
