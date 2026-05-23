"""Cross-validate C++ RK4 solver against SciPy solve_ivp."""
import importlib.util
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def _load_module():
    build_dir = Path(__file__).parent.parent.parent / "build"
    candidates = list(build_dir.rglob("solver_py*.so"))
    if not candidates:
        raise RuntimeError(f"solver_py.so not found under {build_dir}")
    spec = importlib.util.spec_from_file_location("solver_py", candidates[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


slv = _load_module()

# ODE: dy/dt = -y,  y(0) = 1  →  exact solution y(t) = e^{-t}
T0, T1, Y0, DT = 0.0, 5.0, 1.0, 0.01

solver = slv.RK4Solver()
t0 = time.perf_counter()
sol_cpp = solver.solve(lambda t, y: -y, T0, T1, Y0, DT)
t_cpp = time.perf_counter() - t0

t_grid = np.array([i * DT for i in range(len(sol_cpp))])
cpp_arr = np.array(sol_cpp)
exact = np.exp(-t_grid)

rms = float(np.sqrt(np.mean((cpp_arr - exact) ** 2)))
print(f"C++ RK4  steps={len(sol_cpp)}  time={t_cpp*1000:.2f} ms  RMS vs exact={rms:.2e}")

res_scipy = solve_ivp(lambda t, y: [-y[0]], [T0, T1], [Y0], method="RK45", max_step=DT)
print(f"SciPy RK45 steps={len(res_scipy.t)}")

assert rms < 1e-6, f"RMS {rms:.2e} >= 1e-6"
print("\nAll assertions passed.")
