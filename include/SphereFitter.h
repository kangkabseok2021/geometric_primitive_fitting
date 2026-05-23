#pragma once
#include "PointCloud.h"
#include <Eigen/Dense>

struct SphereFitterOptions {
    int    max_iter    = 100;
    double tol         = 1e-8;
    double lambda_init = 1e-3;
    double lambda_max  = 1e8;
    double lambda_min  = 1e-10;
};

struct SphereParams {
    Vec3   center;
    double radius       = 0.0;
    double rms_residual = 0.0;
    int    iterations   = 0;
    bool   converged    = false;
};

class SphereFitter {
public:
    using Options = SphereFitterOptions;

    explicit SphereFitter(Options opts = Options{});

    SphereParams fit(const PointCloud& pts) const;

    static double          residual(const Vec3& p, const Vec3& center, double radius);
    static Eigen::Vector4d jacobian_row(const Vec3& p, const Vec3& center);

private:
    Options opts_;
};
