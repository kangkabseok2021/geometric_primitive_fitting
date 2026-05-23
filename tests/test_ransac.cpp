#include <gtest/gtest.h>
#include "RansacFilter.h"
#include "PointCloud.h"
#include <random>
#include <cmath>

// Test 7: RANSAC rejects outliers effectively
TEST(Ransac, RejectsOutliersEffectively) {
    // 400 clean points + 100 random outliers (20% outlier rate)
    PointCloud pts = generate_noisy_sphere({0,0,0}, 2.0, 400, 0.01, 10);
    std::mt19937 rng(77);
    std::uniform_real_distribution<> uni(-5.0, 5.0);
    for (int i = 0; i < 100; ++i)
        pts.push_back({uni(rng), uni(rng), uni(rng)});

    RansacResult res = ransac_sphere(pts, 0.05, 1000, 42);
    ASSERT_TRUE(res.success);

    double inlier_frac = static_cast<double>(res.inliers.size()) / static_cast<double>(pts.size());
    EXPECT_GE(inlier_frac, 0.70);
    EXPECT_LE(inlier_frac, 0.95);
}

// Test 8: Clean data retains all inliers
TEST(Ransac, CleanDataRetainsAllInliers) {
    PointCloud pts = generate_noisy_sphere({1,2,3}, 3.0, 300, 0.005, 5);
    RansacResult res = ransac_sphere(pts, 0.05, 500, 42);
    ASSERT_TRUE(res.success);
    double inlier_frac = static_cast<double>(res.inliers.size()) / static_cast<double>(pts.size());
    EXPECT_GE(inlier_frac, 0.90);
}

// Test 9: Reproducible with same seed
TEST(Ransac, ReproducibleWithSeed) {
    PointCloud pts = generate_noisy_sphere({0,0,0}, 5.0, 200, 0.02, 1);
    std::mt19937 rng(42);
    std::uniform_real_distribution<> uni(-10.0, 10.0);
    for (int i = 0; i < 40; ++i)
        pts.push_back({uni(rng), uni(rng), uni(rng)});

    RansacResult r1 = ransac_sphere(pts, 0.05, 200, 99);
    RansacResult r2 = ransac_sphere(pts, 0.05, 200, 99);
    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);
    EXPECT_EQ(r1.inliers.size(), r2.inliers.size());
}

// Test 10: Refined center is close to ground truth
TEST(Ransac, RefinedCenterCloseToGroundTruth) {
    Vec3 true_c{2.0, -1.0, 3.0};
    double true_r = 4.0;
    PointCloud pts = generate_noisy_sphere(true_c, true_r, 400, 0.02, 7);
    // Add 10% outliers
    std::mt19937 rng(55);
    std::uniform_real_distribution<> uni(-8.0, 8.0);
    for (int i = 0; i < 40; ++i)
        pts.push_back({uni(rng), uni(rng), uni(rng)});

    RansacResult res = ransac_sphere(pts, 0.1, 1000, 42);
    ASSERT_TRUE(res.success);
    EXPECT_NEAR((res.center - true_c).norm(), 0.0, 0.2);
    EXPECT_NEAR(res.radius, true_r, 0.2);
}
