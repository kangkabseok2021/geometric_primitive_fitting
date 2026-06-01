#pragma once
#include "Detection.h"
#include <vector>
#include <cmath>
#include <numbers>

struct Cluster {
    int   id;
    float cx, cy;        // Cartesian centroid (m)
    float cov[4];        // 2×2 sample covariance row-major
    int   count;
    float snr_avg_dB;
};

class Clusterer {
public:
    // eps: neighbourhood radius in metres; min_pts: core point threshold
    std::vector<Cluster> cluster(const DetectionList& dets,
                                 float eps = 2.0f, int min_pts = 2) const {
        const int N = static_cast<int>(dets.size());
        if (N == 0) return {};

        // Convert to Cartesian
        std::vector<float> xs(N), ys(N);
        for (int i = 0; i < N; ++i) {
            float az_rad = dets[i].azimuth_deg * static_cast<float>(std::numbers::pi) / 180.f;
            xs[i] = dets[i].range_m * std::cos(az_rad);
            ys[i] = dets[i].range_m * std::sin(az_rad);
        }

        std::vector<int> labels(N, -1);  // -1 = noise
        int next_id = 0;

        for (int i = 0; i < N; ++i) {
            if (labels[i] != -1) continue;
            auto nbrs = neighbours(xs, ys, i, eps);
            if ((int)nbrs.size() < min_pts) continue;

            labels[i] = next_id;
            // BFS expansion
            std::vector<int> queue(nbrs.begin(), nbrs.end());
            for (int qi = 0; qi < (int)queue.size(); ++qi) {
                int q = queue[qi];
                if (labels[q] == -1) {
                    labels[q] = next_id;
                    auto qn = neighbours(xs, ys, q, eps);
                    if ((int)qn.size() >= min_pts)
                        for (int nb : qn)
                            if (labels[nb] == -1) queue.push_back(nb);
                } else if (labels[q] < 0) {
                    labels[q] = next_id;
                }
            }
            ++next_id;
        }

        return buildClusters(dets, xs, ys, labels, next_id);
    }

private:
    std::vector<int> neighbours(const std::vector<float>& xs,
                                 const std::vector<float>& ys,
                                 int i, float eps) const {
        std::vector<int> out;
        for (int j = 0; j < (int)xs.size(); ++j) {
            if (j == i) continue;
            float dx = xs[i] - xs[j], dy = ys[i] - ys[j];
            if (std::sqrt(dx*dx + dy*dy) <= eps) out.push_back(j);
        }
        return out;
    }

    std::vector<Cluster> buildClusters(const DetectionList& dets,
                                        const std::vector<float>& xs,
                                        const std::vector<float>& ys,
                                        const std::vector<int>& labels,
                                        int n_clusters) const {
        std::vector<Cluster> out(n_clusters);
        std::vector<int> counts(n_clusters, 0);
        std::vector<float> sx(n_clusters,0), sy(n_clusters,0);
        std::vector<float> snr_sum(n_clusters,0);

        for (int i = 0; i < (int)labels.size(); ++i) {
            int id = labels[i];
            if (id < 0) continue;
            sx[id] += xs[i]; sy[id] += ys[i];
            snr_sum[id] += dets[i].snr_dB;
            ++counts[id];
        }
        for (int id = 0; id < n_clusters; ++id) {
            out[id].id    = id;
            out[id].count = counts[id];
            out[id].cx    = counts[id] > 0 ? sx[id] / counts[id] : 0.f;
            out[id].cy    = counts[id] > 0 ? sy[id] / counts[id] : 0.f;
            out[id].snr_avg_dB = counts[id] > 0 ? snr_sum[id] / counts[id] : 0.f;
        }

        // 2×2 sample covariance
        for (int i = 0; i < (int)labels.size(); ++i) {
            int id = labels[i];
            if (id < 0) continue;
            float dx = xs[i] - out[id].cx, dy = ys[i] - out[id].cy;
            out[id].cov[0] += dx*dx;
            out[id].cov[1] += dx*dy;
            out[id].cov[2] += dx*dy;
            out[id].cov[3] += dy*dy;
        }
        static constexpr float kMinVar = 0.04f;  // 0.2m²
        for (int id = 0; id < n_clusters; ++id) {
            float denom = counts[id] > 1 ? counts[id] - 1.f : 1.f;
            for (float& v : out[id].cov) v /= denom;
            out[id].cov[0] = std::max(out[id].cov[0], kMinVar);
            out[id].cov[3] = std::max(out[id].cov[3], kMinVar);
        }
        return out;
    }
};
