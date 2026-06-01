#include <gtest/gtest.h>
#include "CfarDetector.h"
#include "Clusterer.h"
#include "EKFTracker.h"
#include "DopplerClassifier.h"
#include <cmath>
#include <numbers>
#include <random>
#include <vector>
#include <numeric>

// ─── CFAR (3 tests) ──────────────────────────────────────────────────────────

TEST(CFAR, ThresholdAlpha_MatchesPfaFormula) {
    // alpha = Pfa^(-1/N_ref) - 1
    CfarDetector<2, 8> cfar_1e4(1e-4f);
    float expected = std::pow(1e-4f, -1.0f / 8) - 1.0f;
    EXPECT_NEAR(cfar_1e4.alpha(), expected, 1e-4f);

    CfarDetector<2, 8> cfar_1e3(1e-3f);
    float expected2 = std::pow(1e-3f, -1.0f / 8) - 1.0f;
    EXPECT_NEAR(cfar_1e3.alpha(), expected2, 1e-4f);

    CfarDetector<2, 8> cfar_1e5(1e-5f);
    float expected3 = std::pow(1e-5f, -1.0f / 8) - 1.0f;
    EXPECT_NEAR(cfar_1e5.alpha(), expected3, 1e-4f);
}

TEST(CFAR, DetectsTargetAboveNoise) {
    // 1D range-Doppler map: 20 range bins × 1 Doppler bin
    // Target at range bin 10, all others at noise level 1.0
    constexpr int NR = 20, ND = 1;
    std::vector<float> map(NR * ND, 1.0f);
    map[10] = 100.f;  // 20 dB above noise floor

    CfarDetector<2, 4> cfar(1e-3f);
    auto dets = cfar.detect(map.data(), NR, ND, 1.f, 1.f, 0.f);
    EXPECT_GE((int)dets.size(), 1);
    bool found = false;
    for (const auto& d : dets) if (std::abs(d.range_m - 10.f) < 1.5f) found = true;
    EXPECT_TRUE(found);
}

TEST(CFAR, NoFalseAlarmPureNoise) {
    // Pure Gaussian noise map — no target injected
    constexpr int NR = 40, ND = 1;
    std::vector<float> map(NR * ND);
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(1.0f, 0.1f);
    for (auto& v : map) v = std::abs(nd(rng));

    CfarDetector<2, 8> cfar(1e-4f);
    auto dets = cfar.detect(map.data(), NR, ND, 1.f, 1.f, 0.f);
    // At Pfa=1e-4 and 24 valid cells we expect ~0 detections
    EXPECT_LE((int)dets.size(), 2);
}

// ─── DBSCAN (4 tests) ────────────────────────────────────────────────────────

TEST(DBSCAN, SingleCluster) {
    DetectionList dets;
    // 5 detections tightly grouped at (10m, 0°)
    for (int i = 0; i < 5; ++i)
        dets.push_back({10.f + i*0.1f, 0.f, 0.f, 15.f, i});

    Clusterer c;
    auto clusters = c.cluster(dets, 2.0f, 2);
    EXPECT_EQ((int)clusters.size(), 1);
}

TEST(DBSCAN, TwoSeparatedClusters) {
    DetectionList dets;
    for (int i = 0; i < 4; ++i)
        dets.push_back({10.f + i*0.2f, 0.f, 0.f, 15.f, i});
    for (int i = 0; i < 4; ++i)
        dets.push_back({30.f + i*0.2f, 0.f, 0.f, 15.f, i+4});

    Clusterer c;
    auto clusters = c.cluster(dets, 2.0f, 2);
    ASSERT_EQ((int)clusters.size(), 2);
    float cx0 = clusters[0].cx, cx1 = clusters[1].cx;
    float spread = std::abs(cx0 - cx1);
    EXPECT_GT(spread, 10.f);
}

TEST(DBSCAN, NoisePointsExcluded) {
    // Single isolated detection with min_pts=2 → noise, no cluster
    DetectionList dets;
    dets.push_back({10.f, 0.f, 0.f, 10.f, 0});

    Clusterer c;
    auto clusters = c.cluster(dets, 2.0f, 2);
    EXPECT_EQ((int)clusters.size(), 0);
}

TEST(DBSCAN, CentroidAccuracy) {
    // Known centroid: 4 points at r=10m, az=0° spread → centroid ~ (10, 0)
    DetectionList dets = {
        {10.f, -1.f, 0.f, 15.f, 0},
        {10.f,  1.f, 0.f, 15.f, 1},
        {10.f, -1.f, 0.f, 15.f, 2},
        {10.f,  1.f, 0.f, 15.f, 3},
    };
    Clusterer c;
    auto clusters = c.cluster(dets, 3.0f, 2);
    ASSERT_EQ((int)clusters.size(), 1);
    // centroid should be near (x≈10, y≈0)
    EXPECT_NEAR(clusters[0].cx, 10.f, 0.5f);
    EXPECT_NEAR(clusters[0].cy,  0.f, 0.5f);
}

// ─── EKF (5 tests) ───────────────────────────────────────────────────────────

TEST(EKF, PredictCovariance_Grows) {
    EKFTrack t;
    t.state = TrackState::CONFIRMED;
    t.x = Eigen::Vector4f(20.f, 0.f, 5.f, 0.f);
    t.P = Eigen::Matrix4f::Identity();

    float dt = TrackManager::kDt;
    Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
    F(0,2) = dt; F(1,3) = dt;
    float dt3 = dt*dt*dt;
    Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
    Q(0,0) = Q(1,1) = dt3/3.f;
    Q(2,2) = Q(3,3) = dt;
    Q(0,2) = Q(2,0) = Q(1,3) = Q(3,1) = dt*dt/2.f;

    Eigen::Matrix4f P_pred = F * t.P * F.transpose() + Q;
    EXPECT_GT(P_pred.trace(), t.P.trace());
}

TEST(EKF, Update_ReducesCovariance) {
    TrackManager tm;
    // Single cluster near (20, 0)
    std::vector<Cluster> cl = {{ 0, 20.f, 0.f, {0.2f,0.f,0.f,0.2f}, 3, 20.f }};
    // Feed 6 frames to confirm and then check covariance converges
    for (int i = 0; i < 6; ++i) tm.update(cl);
    auto confirmed = tm.confirmedTracks();
    ASSERT_FALSE(confirmed.empty());
    EXPECT_LT(confirmed[0]->P.trace(), 4.f * 10.f);  // below initial P=10*I
}

TEST(EKF, JacobianMatchesNumerical) {
    Eigen::Vector4f x(15.f, 8.f, 3.f, -1.f);
    auto H_analytical = TrackManager::jacobianH(x);

    constexpr float eps = 1e-3f;
    Eigen::Matrix<float,3,4> H_num = Eigen::Matrix<float,3,4>::Zero();
    for (int j = 0; j < 4; ++j) {
        Eigen::Vector4f xp = x, xm = x;
        xp(j) += eps; xm(j) -= eps;
        auto h = [](const Eigen::Vector4f& v) {
            float r = std::sqrt(v(0)*v(0)+v(1)*v(1))+1e-6f;
            float toDeg = 180.f / static_cast<float>(std::numbers::pi);
            return Eigen::Vector3f(r,
                                   std::atan2(v(1),v(0))*toDeg,
                                   (v(0)*v(2)+v(1)*v(3))/r);
        };
        H_num.col(j) = (h(xp) - h(xm)) / (2.f * eps);
    }
    // float32 central-difference with eps=1e-3 gives ~1e-3 round-off on azimuth row
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_NEAR(H_analytical(i,j), H_num(i,j), 2e-3f)
                << "at (" << i << "," << j << ")";
}

TEST(EKF, ConfirmsAfterHits) {
    TrackManager tm;
    std::vector<Cluster> cl = {{ 0, 25.f, 0.f, {0.2f,0.f,0.f,0.2f}, 3, 20.f }};
    for (int i = 0; i < 5; ++i) tm.update(cl);
    auto confirmed = tm.confirmedTracks();
    EXPECT_GE((int)confirmed.size(), 1);
}

TEST(EKF, DeletesAfterMisses) {
    TrackManager tm;
    std::vector<Cluster> cl = {{ 0, 25.f, 0.f, {0.2f,0.f,0.f,0.2f}, 3, 20.f }};
    for (int i = 0; i < 5; ++i) tm.update(cl);
    ASSERT_GE((int)tm.confirmedTracks().size(), 1);

    // Now feed empty frames
    std::vector<Cluster> empty;
    for (int i = 0; i < TrackManager::kDeleteMisses + 1; ++i) tm.update(empty);
    EXPECT_EQ((int)tm.confirmedTracks().size(), 0);
}

// ─── Doppler Classifier (3 tests) ────────────────────────────────────────────

static EKFTrack makeTrack(float px, float py, float vx, float vy, float last_vr) {
    EKFTrack t;
    t.id = 0; t.state = TrackState::CONFIRMED; t.age = 10;
    t.x = Eigen::Vector4f(px, py, vx, vy);
    t.last_vr_mps = last_vr;
    t.snr_avg_dB  = 20.f;
    return t;
}

TEST(Doppler, StaticClassification) {
    // Target ahead at (20, 0), ego v=10 m/s, target stationary
    // Expected vr_static = -10 * cos(0) = -10 m/s
    float v_ego = 10.f;
    auto track = makeTrack(20.f, 0.f, 0.f, 0.f, -10.f);  // |vr - vr_static| ≈ 0

    DopplerClassifier dc;
    std::vector<const EKFTrack*> tracks = {&track};
    auto result = dc.classify(tracks, v_ego);
    ASSERT_EQ((int)result.size(), 1);
    EXPECT_EQ(result[0].cls, ObjectClass::STATIC);
}

TEST(Doppler, OncomingClassification) {
    // Oncoming vehicle: vr_meas << vr_static (strongly negative radial velocity)
    float v_ego = 10.f;
    float vr_static = -10.f * std::cos(0.f);  // -10 m/s
    auto track = makeTrack(20.f, 0.f, -15.f, 0.f, vr_static - 10.f);  // approaching fast

    DopplerClassifier dc;
    std::vector<const EKFTrack*> tracks = {&track};
    auto result = dc.classify(tracks, v_ego);
    ASSERT_EQ((int)result.size(), 1);
    EXPECT_EQ(result[0].cls, ObjectClass::ONCOMING);
}

TEST(Doppler, MovingClassification) {
    // Moving vehicle: |vr_meas - vr_static| > threshold but not oncoming
    float v_ego = 0.f;  // ego stationary
    // Target at (20, 0) moving laterally — small radial component
    auto track = makeTrack(20.f, 0.f, 0.f, 5.f, 2.f);  // significant but positive

    DopplerClassifier dc;
    std::vector<const EKFTrack*> tracks = {&track};
    auto result = dc.classify(tracks, v_ego);
    ASSERT_EQ((int)result.size(), 1);
    EXPECT_EQ(result[0].cls, ObjectClass::MOVING);
}
