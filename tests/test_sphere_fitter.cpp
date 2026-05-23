#include <gtest/gtest.h>
#include "SphereFitter.h"
#include "PointCloud.h"
#include <cmath>

// Helper: exact sphere surface points (no noise)
static PointCloud exact_sphere(const Vec3& c, double r, int n = 200, unsigned seed = 1) {
    return generate_noisy_sphere(c, r, n, 0.0, seed);
}

// Test 1: Exact fit — zero noise should recover center and radius to tight tolerance
TEST(SphereFitter, ExactFitNoNoise) {
    Vec3 true_c{1.0, -2.0, 3.0};
    double true_r = 5.0;
    PointCloud pts = exact_sphere(true_c, true_r);
    SphereFitter fitter;
    SphereParams result = fitter.fit(pts);
    EXPECT_NEAR((result.center - true_c).norm(), 0.0, 1e-5);
    EXPECT_NEAR(result.radius, true_r, 1e-5);
    EXPECT_NEAR(result.rms_residual, 0.0, 1e-5);
}

// Test 2: Noisy convergence — should recover geometry within noise level
TEST(SphereFitter, NoisyConvergence) {
    Vec3 true_c{0.0, 0.0, 0.0};
    double true_r = 3.0;
    double sigma = 0.02;
    PointCloud pts = generate_noisy_sphere(true_c, true_r, 500, sigma, 7);
    SphereFitter fitter;
    SphereParams result = fitter.fit(pts);
    EXPECT_NEAR((result.center - true_c).norm(), 0.0, 5 * sigma);
    EXPECT_NEAR(result.radius, true_r, 5 * sigma);
    EXPECT_TRUE(result.converged);
}

// Test 3: Analytic Jacobian matches finite differences
TEST(SphereFitter, JacobianMatchesFiniteDiff) {
    Vec3   p{1.0, 2.0, 3.0};
    Vec3   c{0.5, 0.5, 0.5};
    double r = 2.0;
    double eps = 1e-6;

    Eigen::Vector4d J_analytic = SphereFitter::jacobian_row(p, c);

    // Finite differences for parameters [cx, cy, cz, r]
    Eigen::Vector4d J_fd;
    // dc_x
    double f_plus  = SphereFitter::residual(p, c + Vec3{eps,0,0}, r);
    double f_minus = SphereFitter::residual(p, c - Vec3{eps,0,0}, r);
    J_fd(0) = (f_plus - f_minus) / (2 * eps);
    // dc_y
    f_plus  = SphereFitter::residual(p, c + Vec3{0,eps,0}, r);
    f_minus = SphereFitter::residual(p, c - Vec3{0,eps,0}, r);
    J_fd(1) = (f_plus - f_minus) / (2 * eps);
    // dc_z
    f_plus  = SphereFitter::residual(p, c + Vec3{0,0,eps}, r);
    f_minus = SphereFitter::residual(p, c - Vec3{0,0,eps}, r);
    J_fd(2) = (f_plus - f_minus) / (2 * eps);
    // dr
    J_fd(3) = (SphereFitter::residual(p, c, r + eps) -
               SphereFitter::residual(p, c, r - eps)) / (2 * eps);

    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(J_analytic(i), J_fd(i), 1e-5) << "Component " << i;
}

// Test 4: Auto-init — works without user-supplied starting guess
TEST(SphereFitter, AutoInitWorksFromCentroid) {
    Vec3 true_c{10.0, 20.0, 30.0};
    double true_r = 7.5;
    PointCloud pts = generate_noisy_sphere(true_c, true_r, 300, 0.01, 42);
    SphereFitter fitter;
    SphereParams result = fitter.fit(pts);
    EXPECT_NEAR((result.center - true_c).norm(), 0.0, 0.1);
    EXPECT_NEAR(result.radius, true_r, 0.1);
}

// Test 5: Convergence flag set when ||delta|| < tol
TEST(SphereFitter, ConvergenceFlagSetOnSuccess) {
    PointCloud pts = generate_noisy_sphere({0,0,0}, 1.0, 200, 0.005, 3);
    SphereFitter fitter;
    SphereParams result = fitter.fit(pts);
    EXPECT_TRUE(result.converged);
}

// Test 6: RMS residual is consistent with sigma
TEST(SphereFitter, RmsResidualConsistentWithSigma) {
    double sigma = 0.05;
    PointCloud pts = generate_noisy_sphere({0,0,0}, 4.0, 1000, sigma, 99);
    SphereFitter fitter;
    SphereParams result = fitter.fit(pts);
    // RMS residual should be close to the noise sigma
    EXPECT_LT(result.rms_residual, 3.0 * sigma);
    EXPECT_GT(result.rms_residual, 0.0);
}
