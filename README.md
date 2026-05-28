# Numerical Optimization & Geometric Modeling Monorepo

This repository contains multiple advanced computational pipelines built around **C++**, **Eigen**, **nanobind**, and non-linear optimization techniques (like Levenberg-Marquardt), heavily integrated with **Python** for visualization and rapid prototyping.

## Projects in this Repository

### 1. Constrained Skeletal Kinematics Optimizer
A hybrid Python/C++ pipeline that ingests simulated noisy 3D keypoints and mathematically forces them into a biomechanically valid rigid-body skeleton using constrained optimization and DSP techniques.
- **Signal Processing**: 7-sigma-point Unscented Kalman Filter (`filterpy`) + Mahalanobis gating.
- **Optimization**: Levenberg-Marquardt solver (SciPy vs custom C++/Eigen `LDLT`).
- **Acceleration**: C++ implementation achieves >250x speedup over SciPy. Python bindings via `nanobind`.
- **Visualization**: Plotly and Matplotlib 3D overlays.

### 2. Geometric Primitive Fitting via Non-Linear Optimization
A C++17 tool that fits a sphere to noisy 3D point clouds using a custom Levenberg-Marquardt solver with an analytic Jacobian, RANSAC outlier rejection, and a nanobind Python bridge.

### 3. C++23 Compile-Time Numerical Solver
A C++23 numerical solver utilizing `constexpr` for compile-time ODE resolution.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Requires: CMake ≥ 3.20, C++17/23 compiler. Eigen, GoogleTest, and nanobind are fetched automatically.

## Virtual Environment (via `uv`)
To run Python tests, validations, and benchmarks:
```bash
uv venv
source .venv/bin/activate
uv pip install scipy plotly filterpy pytest numpy matplotlib
```

## Running Tests

### C++ Tests (GoogleTest)
```bash
ctest --test-dir build --output-on-failure
```

### Python Tests (PyTest)
```bash
pytest tests/test_signal.py tests/test_pipeline.py
```

## Running Benchmarks (Skeletal Kinematics)
```bash
# Requires the build step to have compiled `skeleton_cpp`
python benchmarks/bench_lm.py
```

## Profiling (Valgrind)
```bash
./scripts/profile.sh
```
Generates `scripts/callgrind.out`. View with `kcachegrind`.
