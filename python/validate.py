"""Cross-validate C++ LM sphere fitter against SciPy least_squares."""
import time
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

# Load the built pybind11 module from the build directory
def _load_module():
    build_dir = Path(__file__).parent.parent / "build"
    candidates = list(build_dir.rglob("sphere_fitter_py*.so"))
    if not candidates:
        raise RuntimeError(f"sphere_fitter_py.so not found under {build_dir}")
    spec = importlib.util.spec_from_file_location("sphere_fitter_py", candidates[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sf = _load_module()

# Ground truth
TRUE_CENTER = np.array([1.5, -0.8, 2.3])
TRUE_RADIUS = 4.2
N_POINTS    = 5000
SIGMA       = 0.02
SEED        = 42

# Generate noisy sphere via C++ helper
pts_cpp = sf.generate_noisy_sphere(TRUE_CENTER, TRUE_RADIUS, N_POINTS, SIGMA, SEED)
pts_np  = np.array(pts_cpp)   # (N, 3)

# --- C++ LM fit ---
fitter = sf.SphereFitter()
t0 = time.perf_counter()
result_cpp = fitter.fit(pts_cpp)
t_cpp = time.perf_counter() - t0

c_cpp = np.array(result_cpp.center)
r_cpp = result_cpp.radius

# --- SciPy LM fit ---
def residuals_scipy(params):
    c = params[:3]
    r = params[3]
    diff = pts_np - c[None, :]
    return np.linalg.norm(diff, axis=1) - r

x0 = np.append(pts_np.mean(axis=0), np.linalg.norm(pts_np - pts_np.mean(axis=0), axis=1).mean())
t0 = time.perf_counter()
res_scipy = least_squares(residuals_scipy, x0, method='lm')
t_scipy = time.perf_counter() - t0

c_scipy = res_scipy.x[:3]
r_scipy = res_scipy.x[3]

# --- Accuracy checks ---
center_err = np.linalg.norm(c_cpp - c_scipy)
radius_err = abs(r_cpp - r_scipy)

print(f"C++   center: {c_cpp}  radius: {r_cpp:.6f}  time: {t_cpp*1000:.2f} ms")
print(f"SciPy center: {c_scipy}  radius: {r_scipy:.6f}  time: {t_scipy*1000:.2f} ms")
print(f"||c_cpp - c_scipy|| = {center_err:.2e}  |r_cpp - r_scipy| = {radius_err:.2e}")
if t_scipy > 0:
    print(f"Speedup: {t_scipy/t_cpp:.1f}x")

TOL = 1e-3
assert center_err < TOL, f"Center error {center_err:.2e} >= {TOL}"
assert radius_err < TOL, f"Radius error {radius_err:.2e} >= {TOL}"
print("\nAll assertions passed.")
