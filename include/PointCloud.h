#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

using Vec3 = Eigen::Vector3d;
using PointCloud = std::vector<Vec3>;

// Load CSV (x,y,z per line). Returns empty on error.
PointCloud load_csv(const std::string& path);

// Generate a noisy sphere point cloud for testing/benchmarking.
// center, radius: true geometry; sigma: Gaussian noise std-dev; seed: RNG seed.
PointCloud generate_noisy_sphere(const Vec3& center, double radius,
                                  int n_points, double sigma, unsigned seed = 42);
