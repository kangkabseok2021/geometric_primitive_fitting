import pytest
import numpy as np
from skeleton_optimizer.pipeline import run_pipeline

def test_pipeline_end_to_end_cpp():
    """Test full pipeline with C++ optimizer"""
    res = run_pipeline(n_frames=50, spike_prob=0.0, use_cpp=True)
    
    assert res['noisy_kp'].shape == (50, 3, 3)
    assert res['smoothed_kp'].shape == (50, 3, 3)
    assert res['theta_seq'].shape == (50, 3)
    
    metrics = res['metrics']
    assert 'peak_elbow_extension_vel_rad_s' in metrics
    assert 'swing_duration_s' in metrics
    assert metrics['swing_duration_s'] > 0

def test_pipeline_end_to_end_py():
    """Test full pipeline with Python optimizer"""
    res = run_pipeline(n_frames=50, spike_prob=0.0, use_cpp=False)
    
    assert res['noisy_kp'].shape == (50, 3, 3)
    assert res['theta_seq'].shape == (50, 3)

def test_equivalence_of_pipeline_results():
    """Test that C++ and Python pipelines produce identical metrics"""
    # Fix seed in simulator implicitly by running same arguments
    # but since noise is random we should mock or just run once and compare outputs
    
    res_cpp = run_pipeline(n_frames=10, spike_prob=0.0, use_cpp=True)
    
    # We must reset seed for simulator to give same sequence!
    # generate_swing_keypoints calls np.random.seed(42) internally!
    res_py = run_pipeline(n_frames=10, spike_prob=0.0, use_cpp=False)
    
    # Check max difference in theta
    diff = np.max(np.abs(res_cpp['theta_seq'] - res_py['theta_seq']))
    assert diff < 1e-4
