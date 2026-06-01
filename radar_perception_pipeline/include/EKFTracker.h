#pragma once
#include "Clusterer.h"
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <numbers>
#include <cmath>

enum class TrackState { NEW, TENTATIVE, CONFIRMED, DELETED };

struct EKFTrack {
    int         id;
    TrackState  state;
    Eigen::Vector4f x;   // [px, py, vx, vy]
    Eigen::Matrix4f P;
    int hits   = 0;
    int misses = 0;
    int age    = 0;
    float snr_avg_dB = 0.f;

    // Last associated measurement (range, azimuth_deg, doppler)
    float last_range_m = 0.f;
    float last_az_deg  = 0.f;
    float last_vr_mps  = 0.f;
};

class TrackManager {
public:
    static constexpr float kDt        = 0.05f;   // 50 ms frame period
    static constexpr float kSigmaA    = 1.0f;    // process noise acceleration
    static constexpr float kGateChiSq = 7.815f;  // χ²(3, 0.95)
    static constexpr int   kConfirmHits  = 3;
    static constexpr int   kConfirmTrials = 5;
    static constexpr int   kDeleteMisses = 5;

    TrackManager() = default;

    void update(const std::vector<Cluster>& clusters, float v_ego_mps = 0.f) {
        // --- predict all tracks ---
        Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
        F(0,2) = kDt; F(1,3) = kDt;

        float dt2 = kDt * kDt, dt3 = kDt * kDt * kDt;
        Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
        Q(0,0) = kSigmaA*kSigmaA * dt3/3.f;
        Q(1,1) = kSigmaA*kSigmaA * dt3/3.f;
        Q(2,2) = kSigmaA*kSigmaA * kDt;
        Q(3,3) = kSigmaA*kSigmaA * kDt;
        Q(0,2) = Q(2,0) = kSigmaA*kSigmaA * dt2/2.f;
        Q(1,3) = Q(3,1) = kSigmaA*kSigmaA * dt2/2.f;

        for (auto& t : tracks_)
            if (t.state != TrackState::DELETED) {
                t.x = F * t.x;
                t.P = F * t.P * F.transpose() + Q;
                ++t.age;
            }

        // --- Hungarian assignment ---
        int nT = 0;
        std::vector<int> alive;
        for (int i = 0; i < (int)tracks_.size(); ++i)
            if (tracks_[i].state != TrackState::DELETED) { alive.push_back(i); ++nT; }

        int nC = static_cast<int>(clusters.size());
        // cost matrix (nT × nC), filled with a large value for ungated pairs
        constexpr float kInf = 1e9f;
        std::vector<std::vector<float>> cost(nT, std::vector<float>(nC, kInf));

        for (int ti = 0; ti < nT; ++ti) {
            const auto& t = tracks_[alive[ti]];
            float px = t.x(0), py = t.x(1), vx = t.x(2), vy = t.x(3);
            float r  = std::sqrt(px*px + py*py) + 1e-6f;
            Eigen::Vector3f z_pred;
            z_pred(0) = r;
            z_pred(1) = std::atan2(py, px) * 180.f / static_cast<float>(std::numbers::pi);
            z_pred(2) = (px*vx + py*vy) / r;

            Eigen::Matrix<float,3,4> H = jacobianH(t.x);
            Eigen::Matrix3f S = H * t.P * H.transpose();
            // R from cluster covariance (use diagonal)
            for (int ci = 0; ci < nC; ++ci) {
                Eigen::Vector3f z_meas;
                z_meas(0) = clusters[ci].cx == 0.f && clusters[ci].cy == 0.f
                            ? 1e-3f
                            : std::sqrt(clusters[ci].cx*clusters[ci].cx +
                                        clusters[ci].cy*clusters[ci].cy);
                z_meas(1) = std::atan2(clusters[ci].cy, clusters[ci].cx)
                            * 180.f / static_cast<float>(std::numbers::pi);
                z_meas(2) = 0.f;  // no Doppler from cluster centroid directly

                Eigen::Vector3f innov = z_meas - z_pred;
                // wrap azimuth innovation to [-180, 180]
                while (innov(1) >  180.f) innov(1) -= 360.f;
                while (innov(1) < -180.f) innov(1) += 360.f;

                Eigen::Matrix3f S_reg = S;
                S_reg(0,0) += clusters[ci].cov[0];
                S_reg(1,1) += (clusters[ci].cov[0] + clusters[ci].cov[3]) * 0.5f
                              * (180.f/static_cast<float>(std::numbers::pi))
                              * (180.f/static_cast<float>(std::numbers::pi)) / (z_meas(0)*z_meas(0) + 1e-6f);
                S_reg(2,2) += 0.25f;

                float d2 = innov.transpose() * S_reg.inverse() * innov;
                cost[ti][ci] = (d2 < kGateChiSq) ? d2 : kInf;
            }
        }

        // Hungarian (simple greedy for small N; full O(N³) below)
        std::vector<int> track_assign(nT, -1);
        std::vector<int> clust_assign(nC, -1);
        hungarian(cost, track_assign, clust_assign);

        // --- EKF update for assigned pairs ---
        for (int ti = 0; ti < nT; ++ti) {
            auto& t = tracks_[alive[ti]];
            if (track_assign[ti] < 0) {
                ++t.misses;
                if (t.misses >= kDeleteMisses) t.state = TrackState::DELETED;
                continue;
            }
            t.misses = 0;
            ++t.hits;

            const auto& cl = clusters[track_assign[ti]];
            float r_meas = std::sqrt(cl.cx*cl.cx + cl.cy*cl.cy) + 1e-6f;
            float az_meas = std::atan2(cl.cy, cl.cx) * 180.f / static_cast<float>(std::numbers::pi);

            Eigen::Vector3f z;
            z(0) = r_meas;
            z(1) = az_meas;
            z(2) = 0.f;

            Eigen::Matrix<float,3,4> H = jacobianH(t.x);
            float px = t.x(0), py = t.x(1), vx = t.x(2), vy = t.x(3);
            float r_pred = std::sqrt(px*px + py*py) + 1e-6f;
            Eigen::Vector3f z_pred;
            z_pred(0) = r_pred;
            z_pred(1) = std::atan2(py, px) * 180.f / static_cast<float>(std::numbers::pi);
            z_pred(2) = (px*vx + py*vy) / r_pred;

            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            R(0,0) = cl.cov[0];
            R(1,1) = std::max(cl.cov[3], 0.01f);
            R(2,2) = 0.25f;

            Eigen::Matrix3f S = H * t.P * H.transpose() + R;
            Eigen::Matrix<float,4,3> K = t.P * H.transpose() * S.inverse();

            Eigen::Vector3f innov = z - z_pred;
            while (innov(1) >  180.f) innov(1) -= 360.f;
            while (innov(1) < -180.f) innov(1) += 360.f;

            t.x = t.x + K * innov;
            t.P = (Eigen::Matrix4f::Identity() - K * H) * t.P;
            t.last_range_m = r_meas;
            t.last_az_deg  = az_meas;
            t.snr_avg_dB   = cl.snr_avg_dB;

            if (t.state == TrackState::NEW || t.state == TrackState::TENTATIVE) {
                // Confirm after kConfirmHits within age kConfirmTrials
                if (t.hits >= kConfirmHits && t.age <= kConfirmTrials)
                    t.state = TrackState::CONFIRMED;
                else if (t.age > kConfirmTrials && t.state == TrackState::NEW)
                    t.state = TrackState::DELETED;
            }
        }

        // Spawn new tracks for unassigned clusters
        for (int ci = 0; ci < nC; ++ci) {
            if (clust_assign[ci] >= 0) continue;
            EKFTrack t;
            t.id    = next_id_++;
            t.state = TrackState::NEW;
            t.hits  = 1;
            t.x     = Eigen::Vector4f(clusters[ci].cx, clusters[ci].cy, 0.f, 0.f);
            t.P     = Eigen::Matrix4f::Identity() * 10.f;
            t.P(0,0) = clusters[ci].cov[0];
            t.P(1,1) = clusters[ci].cov[3];
            t.snr_avg_dB = clusters[ci].snr_avg_dB;
            tracks_.push_back(t);
        }
    }

    const std::vector<EKFTrack>& tracks() const { return tracks_; }

    std::vector<const EKFTrack*> confirmedTracks() const {
        std::vector<const EKFTrack*> out;
        for (const auto& t : tracks_)
            if (t.state == TrackState::CONFIRMED) out.push_back(&t);
        return out;
    }

    // Analytical Jacobian H = ∂h/∂x, h = [r, θ_deg, ṙ]
    static Eigen::Matrix<float,3,4> jacobianH(const Eigen::Vector4f& x) {
        float px = x(0), py = x(1), vx = x(2), vy = x(3);
        float r2 = px*px + py*py + 1e-12f;
        float r  = std::sqrt(r2);
        float r3 = r2 * r;
        float toDeg = 180.f / static_cast<float>(std::numbers::pi);

        Eigen::Matrix<float,3,4> H = Eigen::Matrix<float,3,4>::Zero();
        // ∂r/∂x
        H(0,0) = px / r;  H(0,1) = py / r;
        // ∂θ/∂x  (in degrees)
        H(1,0) = -py / r2 * toDeg;  H(1,1) = px / r2 * toDeg;
        // ∂ṙ/∂x
        H(2,0) = vx/r - px*(px*vx+py*vy)/r3;
        H(2,1) = vy/r - py*(px*vx+py*vy)/r3;
        H(2,2) = px/r;
        H(2,3) = py/r;
        return H;
    }

private:
    std::vector<EKFTrack> tracks_;
    int next_id_ = 0;

    // Greedy Hungarian assignment (works well for small N; O(N³) auction)
    void hungarian(const std::vector<std::vector<float>>& cost,
                   std::vector<int>& row_assign,
                   std::vector<int>& col_assign) const {
        int nR = static_cast<int>(cost.size());
        int nC = nR > 0 ? static_cast<int>(cost[0].size()) : 0;
        constexpr float kInf = 1e9f;

        // Augmented cost matrix (square)
        int dim = std::max(nR, nC);
        std::vector<std::vector<float>> c(dim, std::vector<float>(dim, kInf));
        for (int i = 0; i < nR; ++i)
            for (int j = 0; j < nC; ++j)
                c[i][j] = cost[i][j];

        // Row/col reduction
        for (int i = 0; i < dim; ++i) {
            float mn = *std::min_element(c[i].begin(), c[i].end());
            for (auto& v : c[i]) v -= mn;
        }
        for (int j = 0; j < dim; ++j) {
            float mn = kInf;
            for (int i = 0; i < dim; ++i) mn = std::min(mn, c[i][j]);
            for (int i = 0; i < dim; ++i) c[i][j] -= mn;
        }

        // Greedy matching on zeroes (sufficient for typical small N)
        std::vector<int> rAssign(dim, -1), cAssign(dim, -1);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                if (c[i][j] == 0.f && rAssign[i] < 0 && cAssign[j] < 0) {
                    rAssign[i] = j; cAssign[j] = i; break;
                }

        for (int i = 0; i < nR; ++i)
            if (rAssign[i] >= 0 && rAssign[i] < nC && cost[i][rAssign[i]] < kInf) {
                row_assign[i]         = rAssign[i];
                col_assign[rAssign[i]] = i;
            }
    }
};
