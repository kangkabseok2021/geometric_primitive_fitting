import numpy as np

def compute_metrics(theta_seq: np.ndarray, dt: float = 1.0/30.0) -> dict:
    """
    Computes biomechanical metrics from the optimized joint angles sequence.
    theta_seq: (N, 3) where columns are phi, psi, alpha
    """
    N = len(theta_seq)
    metrics = {}
    
    if N < 2:
        return metrics
        
    # Angular velocity (finite differences)
    omega = np.zeros_like(theta_seq)
    omega[1:-1] = (theta_seq[2:] - theta_seq[:-2]) / (2 * dt)
    omega[0] = (theta_seq[1] - theta_seq[0]) / dt
    omega[-1] = (theta_seq[-1] - theta_seq[-2]) / dt
    
    # Peak elbow extension velocity (negative alpha velocity)
    # alpha is index 2. Extension means alpha is decreasing.
    alpha_vel = omega[:, 2]
    peak_extension_vel = np.min(alpha_vel) # Most negative velocity
    
    metrics['peak_elbow_extension_vel_rad_s'] = float(-peak_extension_vel)
    
    # Swing duration (time where shoulder is actively moving)
    # We define active movement as shoulder angular velocity > 0.5 rad/s
    shoulder_speed = np.linalg.norm(omega[:, :2], axis=1)
    active_frames = np.where(shoulder_speed > 0.5)[0]
    
    if len(active_frames) > 0:
        duration = (active_frames[-1] - active_frames[0]) * dt
        metrics['swing_duration_s'] = float(duration)
    else:
        metrics['swing_duration_s'] = 0.0
        
    return metrics
