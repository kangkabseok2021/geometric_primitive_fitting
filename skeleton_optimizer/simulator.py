import numpy as np
from dataclasses import dataclass

@dataclass
class KeypointFrame:
    timestamp: float
    positions: np.ndarray  # (3, 3) array: [shoulder, elbow, wrist]
    is_spike: bool

def generate_swing_keypoints(n_frames: int = 150, fps: float = 30.0, noise_sigma: float = 0.05, spike_prob: float = 0.05) -> tuple[np.ndarray, np.ndarray, list[KeypointFrame]]:
    """
    Simulates a 3-joint arm executing a simplified golf downswing.
    True trajectory is a sinusoidal arc parameterized by elbow-flexion angle.
    Returns:
        true_kp: (n_frames, 3, 3) true joint positions
        noisy_kp: (n_frames, 3, 3) joint positions with Gaussian noise and spikes
        frames: list of KeypointFrame objects
    """
    np.random.seed(42)  # For reproducible tests
    
    dt = 1.0 / fps
    times = np.arange(n_frames) * dt
    
    L1, L2 = 0.32, 0.28  # Arm lengths in meters
    
    true_kp = np.zeros((n_frames, 3, 3))
    noisy_kp = np.zeros((n_frames, 3, 3))
    frames = []
    
    # Shoulder is fixed at origin
    shoulder = np.array([0.0, 0.0, 0.0])
    
    for i, t in enumerate(times):
        # Parametrize swing: simple sinusoid sweeping elbow flexion
        # Swing takes ~0.5s down, ~0.5s up
        progress = (t % 1.0) * np.pi * 2
        
        # Shoulder azimuth and elevation
        phi = np.radians(45.0 + 30.0 * np.sin(progress))
        psi = np.radians(10.0 + 20.0 * np.cos(progress))
        
        # Elbow flexion (130 degrees down to 15 degrees)
        alpha = np.radians(15.0 + (130.0 - 15.0) * (0.5 * (1 + np.cos(progress))))
        
        # Calculate true positions
        # Direction of upper arm
        ds = np.array([
            np.cos(psi) * np.cos(phi),
            np.cos(psi) * np.sin(phi),
            np.sin(psi)
        ])
        elbow = L1 * ds
        
        # Forearm direction (rotate ds by alpha around perpendicular axis in sagittal plane)
        # Simplified: cross with Z to get a rotation axis
        z_axis = np.array([0.0, 0.0, 1.0])
        rot_axis = np.cross(ds, z_axis)
        rot_axis_norm = np.linalg.norm(rot_axis)
        if rot_axis_norm > 1e-6:
            rot_axis /= rot_axis_norm
        else:
            rot_axis = np.array([1.0, 0.0, 0.0])
            
        # Rodrigues rotation formula
        K = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(alpha) * K + (1 - np.cos(alpha)) * (K @ K)
        
        df = R @ ds
        wrist = elbow + L2 * df
        
        frame_true = np.vstack([shoulder, elbow, wrist])
        true_kp[i] = frame_true
        
        # Add noise
        frame_noisy = frame_true + np.random.normal(0, noise_sigma, size=(3, 3))
        # Keep shoulder fixed even in noisy data
        frame_noisy[0] = shoulder 
        
        # Inject spike outliers
        is_spike = False
        if np.random.random() < spike_prob:
            is_spike = True
            # Add large error to elbow or wrist
            joint_idx = np.random.choice([1, 2])
            frame_noisy[joint_idx] += np.random.uniform(1.0, 2.5, 3)
            
        noisy_kp[i] = frame_noisy
        frames.append(KeypointFrame(timestamp=t, positions=frame_noisy, is_spike=is_spike))
        
    return true_kp, noisy_kp, frames
