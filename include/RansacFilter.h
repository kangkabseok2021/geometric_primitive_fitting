#pragma once
#include "PointCloud.h"
#include "SphereFitter.h"
#include <vector>
#include <cstdint>

struct RansacResult {
    Vec3            center;
    double          radius      = 0.0;
    std::vector<int> inliers;   // indices into input cloud
    bool            success     = false;
};

// 4-point analytical sphere fit: solves 3×3 linear system via Eigen::FullPivLU.
// Returns false (via success flag) if points are degenerate (collinear/coplanar).
RansacResult analytical_fit_4points(const Vec3& p0, const Vec3& p1,
                                     const Vec3& p2, const Vec3& p3);

// RANSAC sphere fit. Runs up to max_iter trials, scores inliers within
// inlier_threshold of the fitted sphere surface. Final model is a full LM
// refit on the consensus inlier set. RNG seeded with `seed` for reproducibility.
RansacResult ransac_sphere(const PointCloud& pts,
                            double inlier_threshold   = 0.05,
                            int    max_iter           = 1000,
                            unsigned seed             = 42,
                            SphereFitter::Options fitter_opts = SphereFitter::Options{});
