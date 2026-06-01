# Numerical Optimization & Geometric Modeling Monorepo

This repository contains multiple C++ and Python pipelines built around **Eigen**, numerical optimization, Kalman filtering, and signal processing — with Python used for validation and visualization.

## Projects

| Project | Language | Description |
|---|---|---|
| **Automotive Radar Perception Pipeline** | C++20 + Python | CA-CFAR detector, DBSCAN clustering, EKF multi-target tracker with analytical Jacobian, Doppler classifier, nlohmann/json object list. 15 GoogleTests + 5 pytest RMSE scenarios. |
| **Constrained Skeletal Kinematics Optimizer** | C++/Python | 7-sigma-point UKF + Levenberg-Marquardt solver fitting 3D keypoints to a rigid articulated skeleton. |
| **Geometric Primitive Fitting** | C++23 | Sphere fitting on noisy 3D point clouds — custom LM solver, analytic Jacobian, RANSAC outlier rejection. |
| **C++23 Numerical Solver** | C++23 | Compile-time ODE solver (Euler + RK4) via `constexpr`, nanobind Python bindings. |

## Build

### Full monorepo
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Requires: CMake ≥ 3.20, C++20 compiler. Eigen 3.4, GoogleTest 1.14, nlohmann/json 3.11.3, and nanobind are fetched automatically via FetchContent.

### Radar perception pipeline (standalone)
```bash
cmake -S radar_perception_pipeline -B build-radar -DCMAKE_BUILD_TYPE=Release
cmake --build build-radar --parallel
ctest --test-dir build-radar --output-on-failure -V
```

Run the Python scenario harness (requires the binary to be built first):
```bash
RADAR_PIPELINE_BIN=build-radar/radar_pipeline pytest radar_perception_pipeline/simulation/ -v
```

## Python environment (via `uv`)
```bash
uv venv && source .venv/bin/activate
uv pip install scipy plotly filterpy pytest numpy matplotlib
```

## Profiling (Valgrind)
```bash
./scripts/profile.sh
```
Generates `scripts/callgrind.out`. View with `kcachegrind`.
