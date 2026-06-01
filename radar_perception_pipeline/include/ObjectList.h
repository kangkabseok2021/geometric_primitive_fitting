#pragma once
#include "DopplerClassifier.h"
#include <nlohmann/json.hpp>
#include <string>

inline std::string classToString(ObjectClass cls) {
    switch (cls) {
        case ObjectClass::STATIC:   return "STATIC";
        case ObjectClass::MOVING:   return "MOVING";
        case ObjectClass::ONCOMING: return "ONCOMING";
        default:                    return "UNKNOWN";
    }
}

inline nlohmann::json toJson(const ClassifiedTrack& t) {
    return {
        {"id",          t.id},
        {"x_m",         t.x_m},
        {"y_m",         t.y_m},
        {"vx_mps",      t.vx_mps},
        {"vy_mps",      t.vy_mps},
        {"heading_deg", t.heading_deg},
        {"class",       classToString(t.cls)},
        {"confidence",  t.confidence},
        {"age_frames",  t.age_frames}
    };
}

inline nlohmann::json toObjectList(const std::vector<ClassifiedTrack>& tracks) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& t : tracks) arr.push_back(toJson(t));
    return arr;
}
