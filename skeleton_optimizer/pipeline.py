import numpy as np
from pathlib import Path
from skeleton_optimizer.simulator import generate_swing_keypoints
from skeleton_optimizer.ukf import filter_joint
from skeleton_optimizer.outlier import filter_sequence
from skeleton_optimizer.kinematics import ArmModel
from skeleton_optimizer.metrics import compute_metrics
import skeleton_optimizer.skeleton_cpp as skeleton_cpp

def run_pipeline(n_frames: int = 150, spike_prob: float = 0.05, noise_sigma: float = 0.05, use_cpp: bool = True) -> dict:
    """
    End-to-end processing pipeline for skeletal kinematics.
    Phase 1: Sim -> UKF -> Outlier Gate
    Phase 2/3: LM Optimization (Python or C++)
    Phase 4: Metrics computation
    """
    model = ArmModel()
    
    # 1. Sim
    _, noisy_kp, _ = generate_swing_keypoints(n_frames, spike_prob=spike_prob, noise_sigma=noise_sigma)
    
    # 2. Gate (detect outliers based on physics limits)
    gated_kp, mask = filter_sequence(noisy_kp)
    
    # 3. UKF (smooth data and estimate uncertainty)
    smoothed_kp = np.zeros_like(noisy_kp)
    weights_seq = np.ones((n_frames, 3))
    
    for joint_idx in range(3):
        # We replace nan from gate with nearest valid or interpolate
        # Simple hold-last-value imputation for filter stability
        imputed_obs = gated_kp[:, joint_idx, :].copy()
        for i in range(1, n_frames):
            if np.isnan(imputed_obs[i, 0]):
                imputed_obs[i] = imputed_obs[i-1]
                
        smoothed, covs = filter_joint(imputed_obs, noise_sigma=noise_sigma)
        smoothed_kp[:, joint_idx, :] = smoothed
        
        # Use determinant of position covariance as inverse confidence
        for i in range(n_frames):
            # Determinant of 3x3 covariance
            det = np.linalg.det(covs[i])
            if mask[i, joint_idx]:
                weights_seq[i, joint_idx] = 1e-4 # Very low confidence if gated
            else:
                # Map det to weight. Small det -> high weight
                weights_seq[i, joint_idx] = 1.0 / (np.sqrt(det) + 1e-6)
                
    # Normalize weights per frame for numerical stability in optimizer
    weights_seq = weights_seq / np.max(weights_seq, axis=1, keepdims=True)
    
    # 4. Constrained LM Optimization
    theta_init = np.array([0.0, np.pi/4, np.pi/2])
    
    if use_cpp:
        # Calls the nanobind C++ extension
        opt_seq = skeleton_cpp.solve_sequence(smoothed_kp, weights_seq, model.L1, model.L2, theta_init, 100)
        theta_seq = opt_seq # Returns (N, 3) directly
    else:
        from skeleton_optimizer.optimizer_py import solve_sequence_py
        results = solve_sequence_py(smoothed_kp, weights_seq, model)
        theta_seq = np.array([r.theta_opt for r in results])
        
    # 5. Metrics
    metrics = compute_metrics(theta_seq)
    
    return {
        'noisy_kp': noisy_kp,
        'smoothed_kp': smoothed_kp,
        'theta_seq': theta_seq,
        'metrics': metrics
    }

if __name__ == "__main__":
    res = run_pipeline()
    print("Pipeline Complete. Metrics:")
    for k, v in res['metrics'].items():
        print(f"  {k}: {v:.2f}")
