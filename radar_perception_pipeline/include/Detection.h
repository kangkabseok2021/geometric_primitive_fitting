#pragma once
#include <vector>

struct Detection {
    float range_m;
    float azimuth_deg;
    float doppler_mps;
    float snr_dB;
    int   target_id;  // ground-truth id from simulator; -1 = clutter
};

using DetectionList = std::vector<Detection>;
