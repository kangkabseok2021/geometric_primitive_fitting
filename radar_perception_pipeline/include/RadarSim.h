#pragma once
#include "Detection.h"
#include <random>
#include <vector>
#include <cmath>
#include <numbers>

static constexpr float kPiF = static_cast<float>(std::numbers::pi);

struct SimTarget {
    float x_m, y_m;
    float vx_mps, vy_mps;
    float rcs_dB;
};

struct SimConfig {
    float sigma_range_m  = 0.1f;
    float sigma_az_deg   = 0.5f;
    float sigma_doppler  = 0.1f;
    float v_ego_mps      = 0.0f;
    float noise_floor_dB = 10.0f;
    unsigned seed        = 42;
};

class RadarSim {
public:
    explicit RadarSim(SimConfig cfg = {}) : cfg_(cfg), rng_(cfg.seed) {}

    DetectionList generate(const std::vector<SimTarget>& targets) {
        DetectionList out;
        std::normal_distribution<float> nr(0.f, cfg_.sigma_range_m);
        std::normal_distribution<float> na(0.f, cfg_.sigma_az_deg);
        std::normal_distribution<float> nd(0.f, cfg_.sigma_doppler);

        for (int i = 0; i < (int)targets.size(); ++i) {
            const auto& t = targets[i];
            float r   = std::sqrt(t.x_m*t.x_m + t.y_m*t.y_m);
            float az  = std::atan2(t.y_m, t.x_m) * 180.f / kPiF;
            // radial Doppler: (v_target - v_ego) projected onto bearing
            float bearing = std::atan2(t.y_m, t.x_m);
            float vr_true = (t.vx_mps * std::cos(bearing) + t.vy_mps * std::sin(bearing))
                          - cfg_.v_ego_mps * std::cos(bearing);
            float snr = t.rcs_dB - cfg_.noise_floor_dB;

            out.push_back({r   + nr(rng_),
                           az  + na(rng_),
                           vr_true + nd(rng_),
                           snr,
                           i});
        }
        return out;
    }

private:
    SimConfig cfg_;
    std::mt19937 rng_;
};
