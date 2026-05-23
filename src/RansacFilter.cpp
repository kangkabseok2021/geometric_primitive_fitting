#include "RansacFilter.h"
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <cmath>

RansacResult analytical_fit_4points(const Vec3& p0, const Vec3& p1,
                                     const Vec3& p2, const Vec3& p3) {
    RansacResult res;
    res.success = false;

    // Subtract p0 to build 3×3 system:
    // For points k=1,2,3: -2Δx·cx - 2Δy·cy - 2Δz·cz = ||p0||² - ||pk||²
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    const Vec3 pts[3] = {p1, p2, p3};
    double d0 = p0.squaredNorm();
    for (int i = 0; i < 3; ++i) {
        Vec3 dp = pts[i] - p0;
        A.row(i) = -2.0 * dp.transpose();
        b(i)     = d0 - pts[i].squaredNorm();
    }

    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
    if (!lu.isInvertible()) return res;

    Vec3   center = lu.solve(b);
    double radius = (p0 - center).norm();
    res.center  = center;
    res.radius  = radius;
    res.success = true;
    return res;
}

RansacResult ransac_sphere(const PointCloud& pts,
                            double inlier_threshold,
                            int    max_iter,
                            unsigned seed,
                            SphereFitter::Options fitter_opts) {
    const int n = static_cast<int>(pts.size());
    RansacResult best;
    best.success = false;

    if (n < 4) return best;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    int best_inlier_count = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Sample 4 distinct indices
        int idx[4];
        for (int i = 0; i < 4; ++i) {
            bool dup;
            do {
                idx[i] = dist(rng);
                dup = false;
                for (int j = 0; j < i; ++j)
                    if (idx[i] == idx[j]) { dup = true; break; }
            } while (dup);
        }

        RansacResult candidate = analytical_fit_4points(
            pts[idx[0]], pts[idx[1]], pts[idx[2]], pts[idx[3]]);
        if (!candidate.success) continue;

        // Score inliers
        std::vector<int> inliers;
        inliers.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            double res = std::abs(SphereFitter::residual(pts[i], candidate.center, candidate.radius));
            if (res < inlier_threshold) inliers.push_back(i);
        }

        if (static_cast<int>(inliers.size()) > best_inlier_count) {
            best_inlier_count = static_cast<int>(inliers.size());
            best.center  = candidate.center;
            best.radius  = candidate.radius;
            best.inliers = inliers;
            best.success = true;

            // Early exit: >90% inliers
            if (inliers.size() > static_cast<size_t>(n) * 9 / 10) break;
        }
    }

    if (!best.success || best.inliers.size() < 4) return best;

    // Final LM refit on consensus set
    PointCloud consensus;
    consensus.reserve(best.inliers.size());
    for (int i : best.inliers) consensus.push_back(pts[i]);

    SphereFitter fitter(fitter_opts);
    try {
        SphereParams refined = fitter.fit(consensus);
        best.center = refined.center;
        best.radius = refined.radius;
        // Recompute inliers after refinement
        best.inliers.clear();
        for (int i = 0; i < n; ++i) {
            if (std::abs(SphereFitter::residual(pts[i], best.center, best.radius)) < inlier_threshold)
                best.inliers.push_back(i);
        }
    } catch (...) {}

    return best;
}
