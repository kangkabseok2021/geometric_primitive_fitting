#include "PointCloud.h"
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>

PointCloud load_csv(const std::string& path) {
    PointCloud pts;
    std::ifstream f(path);
    if (!f.is_open()) return pts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double x, y, z;
        char   comma;
        if (ss >> x >> comma >> y >> comma >> z)
            pts.push_back({x, y, z});
    }
    return pts;
}

PointCloud generate_noisy_sphere(const Vec3& center, double radius,
                                  int n_points, double sigma, unsigned seed) {
    PointCloud pts;
    pts.reserve(static_cast<size_t>(n_points));
    std::mt19937                     rng(seed);
    std::uniform_real_distribution<> azimuth(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<> costheta(-1.0, 1.0);
    std::normal_distribution<>       noise(0.0, sigma);

    for (int i = 0; i < n_points; ++i) {
        double phi   = azimuth(rng);
        double ct    = costheta(rng);
        double st    = std::sqrt(1.0 - ct * ct);
        Vec3 unit{st * std::cos(phi), st * std::sin(phi), ct};
        pts.push_back(center + (radius + noise(rng)) * unit);
    }
    return pts;
}
