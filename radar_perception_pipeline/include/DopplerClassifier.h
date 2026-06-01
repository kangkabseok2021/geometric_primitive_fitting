#pragma once
#include "EKFTracker.h"
#include <vector>
#include <cmath>
#include <numbers>
#include <unordered_map>

enum class ObjectClass { STATIC, MOVING, ONCOMING, UNKNOWN };

struct ClassifiedTrack {
    int         id;
    float       x_m, y_m;
    float       vx_mps, vy_mps;
    float       heading_deg;
    ObjectClass cls;
    float       confidence;
    int         age_frames;
};

class DopplerClassifier {
public:
    static constexpr float kThreshold  = 0.5f;    // m/s
    static constexpr float kLogOddsInc = 0.85f;
    static constexpr int   kGridCells  = 200;
    static constexpr float kGridRes    = 0.5f;     // m per cell

    DopplerClassifier() : grid_(kGridCells * kGridCells, 0.f) {}

    std::vector<ClassifiedTrack> classify(const std::vector<const EKFTrack*>& tracks,
                                          float v_ego_mps) {
        std::vector<ClassifiedTrack> out;
        for (const auto* t : tracks) {
            float px = t->x(0), py = t->x(1);
            float vx = t->x(2), vy = t->x(3);
            float bearing = std::atan2(py, px);

            // Expected radial velocity if stationary
            float vr_static = -v_ego_mps * std::cos(bearing);
            float vr_meas   = t->last_vr_mps;

            ObjectClass cls;
            if (std::abs(vr_meas - vr_static) < kThreshold) {
                cls = ObjectClass::STATIC;
                updateOccupancy(px, py);
            } else if (vr_meas < vr_static - kThreshold) {
                cls = ObjectClass::ONCOMING;
            } else {
                cls = ObjectClass::MOVING;
            }

            float snr_norm = std::max(0.f, t->snr_avg_dB / 30.f);
            float confidence = 1.f / (1.f + std::exp(-5.f * (snr_norm - 0.5f)));

            float heading_deg = std::atan2(vy, vx) * 180.f / static_cast<float>(std::numbers::pi);

            out.push_back({t->id, px, py, vx, vy, heading_deg, cls, confidence, t->age});
        }
        return out;
    }

    float gridOccupancy(int row, int col) const {
        if (row < 0 || row >= kGridCells || col < 0 || col >= kGridCells) return 0.f;
        return grid_[row * kGridCells + col];
    }

private:
    std::vector<float> grid_;

    void updateOccupancy(float x, float y) {
        int col = static_cast<int>((x + kGridCells * kGridRes / 2.f) / kGridRes);
        int row = static_cast<int>((y + kGridCells * kGridRes / 2.f) / kGridRes);
        if (row >= 0 && row < kGridCells && col >= 0 && col < kGridCells)
            grid_[row * kGridCells + col] += kLogOddsInc;
    }
};
