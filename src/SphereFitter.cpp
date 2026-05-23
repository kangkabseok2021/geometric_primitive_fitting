#include "SphereFitter.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

SphereFitter::SphereFitter(Options opts) : opts_(opts) {}

double SphereFitter::residual(const Vec3& p, const Vec3& center, double radius) {
    return (p - center).norm() - radius;
}

Eigen::Vector4d SphereFitter::jacobian_row(const Vec3& p, const Vec3& center) {
    Vec3   diff = p - center;
    double dist = diff.norm();
    if (dist < 1e-12) return Eigen::Vector4d::Zero();
    Eigen::Vector4d J;
    J.head<3>() = -diff / dist;
    J(3)        = -1.0;
    return J;
}

SphereParams SphereFitter::fit(const PointCloud& pts) const {
    if (pts.size() < 4)
        throw std::invalid_argument("Need at least 4 points to fit a sphere");

    // Auto-initialise: centroid + mean distance
    Vec3 c = Vec3::Zero();
    for (const auto& p : pts) c += p;
    c /= static_cast<double>(pts.size());

    double r = 0.0;
    for (const auto& p : pts) r += (p - c).norm();
    r /= static_cast<double>(pts.size());

    const int    n   = static_cast<int>(pts.size());
    double       lam = opts_.lambda_init;
    SphereParams result;
    result.converged = false;

    for (int iter = 0; iter < opts_.max_iter; ++iter) {
        // Build J (n×4) and f (n×1)
        Eigen::MatrixXd J(n, 4);
        Eigen::VectorXd f(n);
        for (int i = 0; i < n; ++i) {
            f(i)    = residual(pts[i], c, r);
            J.row(i) = jacobian_row(pts[i], c).transpose();
        }

        // Normal equations: (J'J + λ·diag(J'J))δ = -J'f
        Eigen::Matrix4d JtJ  = J.transpose() * J;
        Eigen::Vector4d Jtf  = J.transpose() * f;
        Eigen::Matrix4d damp = (lam * JtJ.diagonal()).asDiagonal();
        Eigen::Matrix4d A    = JtJ + damp;

        Eigen::Vector4d delta = A.ldlt().solve(-Jtf);

        double cost_before = f.squaredNorm();

        Vec3   c_new = c + delta.head<3>();
        double r_new = r + delta(3);

        // Compute new cost
        double cost_after = 0.0;
        for (int i = 0; i < n; ++i) {
            double res = residual(pts[i], c_new, r_new);
            cost_after += res * res;
        }

        if (cost_after < cost_before) {
            c   = c_new;
            r   = r_new;
            lam = std::max(opts_.lambda_min, lam / 10.0);
        } else {
            lam = std::min(opts_.lambda_max, lam * 10.0);
        }

        result.iterations = iter + 1;
        if (delta.norm() < opts_.tol) {
            result.converged = true;
            break;
        }
    }

    // Final RMS residual
    double sse = 0.0;
    for (const auto& p : pts) {
        double res = residual(p, c, r);
        sse += res * res;
    }
    result.center       = c;
    result.radius       = r;
    result.rms_residual = std::sqrt(sse / static_cast<double>(pts.size()));
    return result;
}
