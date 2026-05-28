# Numerical Optimization & Geometric Modeling Monorepo

This repository contains multiple advanced computational pipelines built around **C++**, **Eigen**, **nanobind**, and non-linear optimization techniques (like Levenberg-Marquardt), heavily integrated with **Python** for visualization and rapid prototyping.

## Projects in this Repository

| Project | Description | Docs |
|---|---|---|
| **Constrained Skeletal Kinematics Optimizer** | Hybrid Python/C++ pipeline combining a 7-sigma-point Unscented Kalman Filter and a custom C++/Eigen Levenberg-Marquardt solver to fit simulated 3D keypoints to a rigid articulated skeleton. | [docs/MATH.md](docs/MATH.md)<br>[docs/BENCHMARK.md](docs/BENCHMARK.md) |
| **Geometric Primitive Fitting** | C++23 tool fitting a sphere to noisy 3D point clouds using a custom Levenberg-Marquardt solver, analytic Jacobian, and RANSAC outlier rejection. | N/A |
| **C++23 Numerical Solver** | C++23 compile-time numerical ODE solver utilizing `constexpr`. | N/A |

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Requires: CMake ≥ 3.20, C++23 compiler. Eigen, GoogleTest, and nanobind are fetched automatically.

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
