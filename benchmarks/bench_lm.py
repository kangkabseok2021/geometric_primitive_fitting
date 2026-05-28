import timeit
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from skeleton_optimizer.kinematics import ArmModel
from skeleton_optimizer.optimizer_py import solve_frame_py, solve_sequence_py
from skeleton_optimizer.simulator import generate_swing_keypoints
import skeleton_optimizer.skeleton_cpp as skeleton_cpp

def main():
    model = ArmModel()
    true_kp, noisy_kp, _ = generate_swing_keypoints(n_frames=150)
    
    # Use uniform weights for bench
    w = np.array([1e6, 1.0, 1.0])
    w_seq = np.tile(w, (150, 1))
    
    obs = noisy_kp[75]
    theta0 = np.array([0.0, np.pi/4, np.pi/2])
    
    print("Validating equivalence...")
    py_res = solve_frame_py(obs, w, model, theta0).theta_opt
    cpp_res = skeleton_cpp.solve_lm(obs, w, model.L1, model.L2, theta0).theta
    
    diff = np.linalg.norm(py_res - cpp_res)
    assert diff < 1e-5, f"Mismatch between C++ and Python: {diff}"
    print(f"Equivalence verified (diff: {diff:.2e})")
    
    print("\nBenchmarking Single Frame...")
    n_runs = 100
    
    t_py = timeit.timeit(lambda: solve_frame_py(obs, w, model, theta0), number=n_runs) / n_runs
    t_cpp = timeit.timeit(lambda: skeleton_cpp.solve_lm(obs, w, model.L1, model.L2, theta0), number=n_runs) / n_runs
    
    print(f"Python (SciPy): {t_py*1000:.3f} ms / frame")
    print(f"C++ (Eigen)   : {t_cpp*1000:.3f} ms / frame")
    print(f"Speedup       : {t_py/t_cpp:.1f}x")
    
    assert t_cpp < t_py / 20.0, "C++ version should be at least 20x faster"
    
    print("\nBenchmarking Sequence (150 frames)...")
    t_seq_py = timeit.timeit(lambda: solve_sequence_py(noisy_kp, w_seq, model), number=10) / 10
    t_seq_cpp = timeit.timeit(lambda: skeleton_cpp.solve_sequence(noisy_kp, w_seq, model.L1, model.L2, theta0), number=10) / 10
    
    print(f"Python (SciPy): {t_seq_py*1000:.1f} ms")
    print(f"C++ (Eigen)   : {t_seq_cpp*1000:.1f} ms")
    print(f"Speedup       : {t_seq_py/t_seq_cpp:.1f}x")
    
    # Save benchmark results
    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    with open(docs_dir / "BENCHMARK.md", "w") as f:
        f.write("# Constrained Optimizer Benchmarks\n\n")
        f.write("Comparison of SciPy `least_squares` versus native C++ (Eigen LDLT + pybind11/nanobind).\n\n")
        f.write("| Solver | Single Frame | 150 Frame Sequence | Speedup |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Python (SciPy) | {t_py*1000:.3f} ms | {t_seq_py*1000:.1f} ms | 1x |\n")
        f.write(f"| C++ (Eigen) | {t_cpp*1000:.3f} ms | {t_seq_cpp*1000:.1f} ms | **{t_py/t_cpp:.1f}x** |\n")

if __name__ == "__main__":
    main()
