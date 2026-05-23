# Geometric Primitive Fitting via Non-Linear Optimization

C++17 tool that fits a sphere to noisy 3D point clouds (simulating coordinate measuring machine data) using a **custom Levenberg-Marquardt solver** with an **analytic Jacobian**, **RANSAC outlier rejection**, and a **pybind11 Python bridge** cross-validated against SciPy.

## Key Technical Highlights

| What | How |
|------|-----|
| LM solver | Marquardt damping `λ·diag(JᵀJ)` — scale-invariant, no external solver |
| Analytic Jacobian | `Jᵢ = [−(pᵢ−c)/‖pᵢ−c‖, −1]` — verified by finite-difference test |
| Normal equations | Eigen `LDLT` on 4×4 system — numerically stable for PSD matrices |
| RANSAC | 4-point analytical fit via `Eigen::FullPivLU`, LM refit on consensus set |
| Python bridge | pybind11 — zero-copy NumPy↔Eigen, `scipy.optimize.least_squares` cross-validation |
| Profiling | Valgrind Callgrind on 1M points (`scripts/profile.sh`) |

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Requires: CMake ≥ 3.20, C++17 compiler. Eigen, GoogleTest, pybind11 are fetched automatically.

## Run Tests (10 GoogleTests)

```bash
ctest --test-dir build --output-on-failure
```

Tests cover: exact fit, noisy convergence, Jacobian finite-difference check, RANSAC outlier rejection, reproducibility, ground-truth recovery.

## Python Cross-Validation

```bash
# Build the Python module first
cmake --build build --target sphere_fitter_py

# Run validation (requires numpy, scipy)
pip install numpy scipy
python3 python/validate.py
```

Expected output: `‖c_cpp − c_scipy‖ < 1e-3`, speedup reported.

## Profiling (Valgrind)

```bash
# Requires: valgrind, g++
./scripts/profile.sh
```

Generates `scripts/callgrind.out`. View with `kcachegrind`.

## Project Structure

```
include/
  PointCloud.h        — Vec3, PointCloud typedef, generators
  SphereFitter.h      — LM fitter interface + SphereFitterOptions
  RansacFilter.h      — RANSAC interface + RansacResult
src/
  PointCloud.cpp      — CSV loader + noisy sphere generator
  SphereFitter.cpp    — LM solver, analytic Jacobian, LDLT
  RansacFilter.cpp    — 4-point analytical fit, RANSAC loop
python/
  bindings.cpp        — pybind11 module
  validate.py         — SciPy cross-validation
tests/
  test_sphere_fitter.cpp  — 6 LM solver tests
  test_ransac.cpp         — 4 RANSAC tests
scripts/
  profile.sh          — Valgrind Callgrind profiling on 1M points
```
