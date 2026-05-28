# Constrained Optimizer Benchmarks

Comparison of SciPy `least_squares` versus native C++ (Eigen LDLT + pybind11/nanobind).

| Solver | Single Frame | 150 Frame Sequence | Speedup |
|---|---|---|---|
| Python (SciPy) | 2.458 ms | 261.3 ms | 1x |
| C++ (Eigen) | 0.011 ms | 0.9 ms | **222.2x** |
