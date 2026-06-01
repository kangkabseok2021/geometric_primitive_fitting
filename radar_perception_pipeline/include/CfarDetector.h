#pragma once
#include "Detection.h"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// CA-CFAR with OS-CFAR fallback for clutter edges.
// N_guard: one-sided guard cells; N_ref: one-sided reference cells.
template<int N_guard = 2, int N_ref = 8>
class CfarDetector {
public:
    static constexpr int kWindow = 2 * N_guard + 2 * N_ref + 1;

    // target_pfa: desired false-alarm probability per range-Doppler cell
    explicit CfarDetector(float target_pfa = 1e-4f)
        : alpha_(std::pow(target_pfa, -1.0f / N_ref) - 1.0f) {}

    // rd_map: row-major float[n_range * n_doppler], power values (linear)
    // Returns detections in (range-bin, doppler-bin) index space.
    // Each output Detection: range_m = range_bin, azimuth_deg = doppler_bin (re-interpreted by caller)
    DetectionList detect(const float* rd_map, int n_range, int n_doppler,
                         float range_res_m = 1.0f, float doppler_res_mps = 1.0f,
                         float az_deg = 0.0f) const {
        DetectionList dets;
        for (int ri = N_guard + N_ref; ri < n_range - N_guard - N_ref; ++ri) {
            for (int di = 0; di < n_doppler; ++di) {
                float cell = rd_map[ri * n_doppler + di];
                // Collect reference cells (1D along range axis, fixed Doppler bin)
                std::vector<float> ref;
                ref.reserve(2 * N_ref);
                for (int k = ri - N_guard - N_ref; k < ri - N_guard; ++k)
                    ref.push_back(rd_map[k * n_doppler + di]);
                for (int k = ri + N_guard + 1; k <= ri + N_guard + N_ref; ++k)
                    ref.push_back(rd_map[k * n_doppler + di]);

                float threshold = computeThreshold(ref);
                if (cell > threshold) {
                    float snr_dB = 10.f * std::log10(cell / (threshold / alpha_ + 1e-12f));
                    dets.push_back({ri * range_res_m, az_deg,
                                    di * doppler_res_mps, snr_dB, -1});
                }
            }
        }
        return dets;
    }

    float alpha() const { return alpha_; }

private:
    float alpha_;

    float computeThreshold(std::vector<float>& ref) const {
        float mu  = std::accumulate(ref.begin(), ref.end(), 0.0f) / ref.size();
        float mu2 = 0.f;
        for (float v : ref) mu2 += v * v;
        mu2 /= ref.size();
        float variance = mu2 - mu * mu;

        // OS-CFAR fallback: heterogeneous clutter (variance > 3*mean²)
        if (variance > 3.f * mu * mu) {
            std::sort(ref.begin(), ref.end());
            // k-th ordered statistic: use 3/4 of reference cells
            int k = static_cast<int>(ref.size() * 3 / 4);
            return alpha_ * ref[k];
        }
        return alpha_ * mu;
    }
};
