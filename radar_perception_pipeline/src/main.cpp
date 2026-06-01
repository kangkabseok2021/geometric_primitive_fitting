#include "RadarSim.h"
#include "CfarDetector.h"
#include "Clusterer.h"
#include "EKFTracker.h"
#include "DopplerClassifier.h"
#include "ObjectList.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>

// Runs the five-stage pipeline on a scenario JSON file and writes the object list.
// Usage: radar_pipeline --scenario <input.json> --output <output.json>
int main(int argc, char* argv[]) {
    std::string scenario_path, output_path;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--scenario") scenario_path = argv[i+1];
        if (std::string(argv[i]) == "--output")   output_path   = argv[i+1];
    }
    if (scenario_path.empty() || output_path.empty()) {
        std::cerr << "Usage: radar_pipeline --scenario <in.json> --output <out.json>\n";
        return 1;
    }

    nlohmann::json scenario;
    {
        std::ifstream f(scenario_path);
        if (!f) { std::cerr << "Cannot open " << scenario_path << "\n"; return 1; }
        f >> scenario;
    }

    float v_ego = scenario.value("v_ego_mps", 0.f);
    SimConfig cfg;
    cfg.v_ego_mps = v_ego;

    std::vector<SimTarget> targets;
    for (const auto& t : scenario["targets"]) {
        targets.push_back({t["x_m"], t["y_m"],
                           t.value("vx_mps", 0.f), t.value("vy_mps", 0.f),
                           t.value("rcs_dB", 20.f)});
    }

    int n_frames = scenario.value("n_frames", 20);

    RadarSim       sim(cfg);
    CfarDetector<> cfar;
    Clusterer      clusterer;
    TrackManager   tracker;
    DopplerClassifier classifier;

    for (int frame = 0; frame < n_frames; ++frame) {
        // Simulate detections directly (point-cloud mode — no range-Doppler map)
        DetectionList dets = sim.generate(targets);

        // DBSCAN clustering (skip CFAR map stage in point-cloud simulation mode)
        auto clusters = clusterer.cluster(dets, 2.0f, 2);

        // EKF update
        tracker.update(clusters, v_ego);

        // Simple motion: constant velocity
        for (auto& t : targets) { t.x_m += t.vx_mps * 0.05f; t.y_m += t.vy_mps * 0.05f; }
    }

    auto confirmed = tracker.confirmedTracks();
    std::vector<ClassifiedTrack> classified = classifier.classify(confirmed, v_ego);
    auto obj_list = toObjectList(classified);

    std::ofstream out(output_path);
    if (!out) { std::cerr << "Cannot write " << output_path << "\n"; return 1; }
    out << obj_list.dump(2) << "\n";
    return 0;
}
