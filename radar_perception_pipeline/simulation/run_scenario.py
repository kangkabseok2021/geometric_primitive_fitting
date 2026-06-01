"""
pytest harness: generates ground-truth scenarios, invokes radar_pipeline binary,
parses object list JSON, asserts RMSE < 0.5 m and FPR < 5%.
"""
import json
import math
import os
import subprocess
import tempfile
import pytest

BINARY = os.environ.get(
    "RADAR_PIPELINE_BIN",
    os.path.join(os.path.dirname(__file__), "..", "build", "radar_perception_pipeline", "radar_pipeline"),
)

def run_pipeline(scenario: dict) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as sf:
        json.dump(scenario, sf)
        sf_path = sf.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as of:
        of_path = of.name

    try:
        result = subprocess.run(
            [BINARY, "--scenario", sf_path, "--output", of_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Binary failed: {result.stderr}")
        with open(of_path) as fh:
            return json.load(fh)
    finally:
        os.unlink(sf_path)
        os.unlink(of_path)


def rmse(tracks: list[dict], targets: list[dict]) -> float:
    if not tracks or not targets:
        return float("inf")
    errors = []
    for tgt in targets:
        best = min(
            math.sqrt((t["x_m"] - tgt["x_m"]) ** 2 + (t["y_m"] - tgt["y_m"]) ** 2)
            for t in tracks
        )
        errors.append(best)
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def fpr(tracks: list[dict], targets: list[dict], gate_m: float = 5.0) -> float:
    if not tracks:
        return 0.0
    false_positives = sum(
        1
        for t in tracks
        if all(
            math.sqrt((t["x_m"] - tgt["x_m"]) ** 2 + (t["y_m"] - tgt["y_m"]) ** 2) > gate_m
            for tgt in targets
        )
    )
    return false_positives / len(tracks) if tracks else 0.0


# ─── Scenarios ───────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "single_static_target",
        "v_ego_mps": 0.0,
        "n_frames": 25,
        "targets": [{"x_m": 30.0, "y_m": 0.0, "vx_mps": 0.0, "vy_mps": 0.0, "rcs_dB": 20.0}],
    },
    {
        "name": "two_moving_targets",
        "v_ego_mps": 0.0,
        "n_frames": 25,
        "targets": [
            {"x_m": 20.0, "y_m":  5.0, "vx_mps": 2.0, "vy_mps": 0.0, "rcs_dB": 20.0},
            {"x_m": 20.0, "y_m": -5.0, "vx_mps": 2.0, "vy_mps": 0.0, "rcs_dB": 20.0},
        ],
    },
    {
        "name": "oncoming_target_with_ego",
        "v_ego_mps": 10.0,
        "n_frames": 25,
        "targets": [{"x_m": 50.0, "y_m": 0.0, "vx_mps": -10.0, "vy_mps": 0.0, "rcs_dB": 22.0}],
    },
    {
        "name": "four_targets_mixed",
        "v_ego_mps": 5.0,
        "n_frames": 30,
        "targets": [
            {"x_m": 25.0, "y_m":  3.0, "vx_mps":  0.0, "vy_mps": 0.0, "rcs_dB": 20.0},
            {"x_m": 40.0, "y_m": -3.0, "vx_mps": -5.0, "vy_mps": 0.0, "rcs_dB": 20.0},
            {"x_m": 15.0, "y_m":  8.0, "vx_mps":  3.0, "vy_mps": 1.0, "rcs_dB": 18.0},
            {"x_m": 35.0, "y_m": -8.0, "vx_mps":  1.0, "vy_mps": 2.0, "rcs_dB": 18.0},
        ],
    },
    {
        "name": "closely_spaced_targets",
        "v_ego_mps": 0.0,
        "n_frames": 30,
        "targets": [
            {"x_m": 20.0, "y_m":  1.5, "vx_mps": 1.0, "vy_mps":  0.5, "rcs_dB": 20.0},
            {"x_m": 20.0, "y_m": -1.5, "vx_mps": 1.0, "vy_mps": -0.5, "rcs_dB": 20.0},
            {"x_m": 30.0, "y_m":  0.0, "vx_mps": 0.0, "vy_mps":  0.0, "rcs_dB": 22.0},
        ],
    },
]


def _skip_if_no_binary():
    if not os.path.isfile(BINARY):
        pytest.skip(f"radar_pipeline binary not found at {BINARY}")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_scenario_rmse_and_fpr(scenario):
    _skip_if_no_binary()
    tracks = run_pipeline(scenario)
    targets = scenario["targets"]

    r = rmse(tracks, targets)
    fp = fpr(tracks, targets)

    assert r < 0.5, f"[{scenario['name']}] RMSE={r:.3f} m >= 0.5 m"
    assert fp < 0.05, f"[{scenario['name']}] FPR={fp:.2%} >= 5%"
