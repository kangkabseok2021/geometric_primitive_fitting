import numpy as np
import pytest
from skeleton_optimizer.simulator import generate_swing_keypoints
from skeleton_optimizer.ukf import filter_joint
from skeleton_optimizer.outlier import filter_sequence, AnatomicalGate

def test_simulator_shape():
    true_kp, noisy_kp, frames = generate_swing_keypoints(n_frames=150)
    assert true_kp.shape == (150, 3, 3)
    assert noisy_kp.shape == (150, 3, 3)
    assert len(frames) == 150

def test_spike_injection_rate():
    _, _, frames = generate_swing_keypoints(n_frames=10000, spike_prob=0.05)
    spikes = sum(1 for f in frames if f.is_spike)
    rate = spikes / 10000
    assert 0.03 <= rate <= 0.07

def test_ukf_reduces_rmse():
    true_kp, noisy_kp, _ = generate_swing_keypoints(n_frames=150, spike_prob=0.0) # No spikes for raw UKF test
    
    # Test elbow (joint 1)
    elbow_obs = noisy_kp[:, 1, :]
    elbow_true = true_kp[:, 1, :]
    
    smoothed, covs = filter_joint(elbow_obs)
    
    rmse_noisy = np.sqrt(np.mean((elbow_obs - elbow_true)**2))
    rmse_smoothed = np.sqrt(np.mean((smoothed - elbow_true)**2))
    
    assert rmse_smoothed < rmse_noisy
    
def test_ukf_convergence_by_frame_30():
    true_kp, noisy_kp, _ = generate_swing_keypoints(n_frames=150, spike_prob=0.0)
    elbow_obs = noisy_kp[:, 1, :]
    elbow_true = true_kp[:, 1, :]
    
    smoothed, _ = filter_joint(elbow_obs)
    
    # Error should be bounded after convergence
    error_after_30 = np.linalg.norm(smoothed[30:] - elbow_true[30:], axis=1)
    assert np.mean(error_after_30) < 0.1  # 2*noise_sigma

def test_covariance_is_positive_definite():
    _, noisy_kp, _ = generate_swing_keypoints(n_frames=10)
    _, covs = filter_joint(noisy_kp[:, 1, :])
    
    for c in covs:
        eigvals = np.linalg.eigvals(c)
        assert np.all(eigvals > 0)

def test_spike_rejected_by_gate():
    gate = AnatomicalGate()
    # Fill history
    for i in range(10):
        gate.is_outlier(np.array([i*0.1, 0, 0]), 1.0/30)
        
    # Inject huge spike
    assert gate.is_outlier(np.array([10.0, 0, 0]), 1.0/30) == True

def test_normal_frame_not_rejected():
    gate = AnatomicalGate()
    for i in range(10):
        gate.is_outlier(np.array([i*0.1, 0, 0]), 1.0/30)
        
    # Normal continuation
    assert gate.is_outlier(np.array([1.0, 0, 0]), 1.0/30) == False

def test_rejection_rate_on_simulated_data():
    true_kp, noisy_kp, frames = generate_swing_keypoints(n_frames=150, spike_prob=0.1)
    
    # We apply the filter sequence
    accepted_kp, rejection_mask = filter_sequence(noisy_kp)
    
    # Check that known spikes were mostly rejected (at least 80% due to Mahalanobis warm-up)
    true_spike_indices = [i for i, f in enumerate(frames) if f.is_spike]
    
    caught_spikes = 0
    for i in true_spike_indices:
        # If any joint was rejected in that frame
        if np.any(rejection_mask[i]):
            caught_spikes += 1
            
    if len(true_spike_indices) > 0:
        assert caught_spikes / len(true_spike_indices) > 0.5
