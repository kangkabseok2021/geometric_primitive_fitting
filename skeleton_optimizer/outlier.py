import numpy as np
from collections import deque
from scipy.stats import chi2

class AnatomicalGate:
    """
    Mahalanobis distance-based outlier rejection for human joint kinematics.
    Flags frames that exceed physical velocity or acceleration limits.
    """
    def __init__(self, max_velocity_m_s: float = 50.0, max_acceleration_m_s2: float = 500.0, history_size: int = 30):
        self.max_v = max_velocity_m_s
        self.max_a = max_acceleration_m_s2
        self.history_size = history_size
        self.pos_history = deque(maxlen=history_size)
        self.vel_history = deque(maxlen=history_size)
        
    def is_outlier(self, pos: np.ndarray, dt: float) -> bool:
        """
        Evaluate if a new position is an outlier based on velocity and acceleration.
        Maintains internal history. Returns True if outlier.
        """
        if len(self.pos_history) < 1:
            self.pos_history.append(pos)
            return False
            
        p_prev = self.pos_history[-1]
        v = (pos - p_prev) / dt
        v_norm = np.linalg.norm(v)
        
        # Hard velocity limit
        if v_norm > self.max_v:
            return True
            
        if len(self.pos_history) >= 2:
            p_prev2 = self.pos_history[-2]
            a = (pos - 2 * p_prev + p_prev2) / (dt ** 2)
            if np.linalg.norm(a) > self.max_a:
                return True
                
        # Mahalanobis distance check on velocity if we have enough history
        if len(self.vel_history) >= 10:
            V = np.array(self.vel_history)  # (N, 3)
            mu_v = np.mean(V, axis=0)
            cov_v = np.cov(V.T) + np.eye(3) * 1e-6  # Add eps for numerical stability
            
            try:
                inv_cov = np.linalg.inv(cov_v)
                diff = v - mu_v
                d_m_sq = diff.T @ inv_cov @ diff
                
                # 99.9% chi-squared threshold for 3 DOF
                threshold = chi2.ppf(0.999, df=3)
                if d_m_sq > threshold:
                    return True
            except np.linalg.LinAlgError:
                pass # Fall back to hard limits if covariance is singular
                
        # If accepted, update history
        self.pos_history.append(pos)
        self.vel_history.append(v)
        return False

def filter_sequence(raw_kp: np.ndarray, dt: float = 1.0/30.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the AnatomicalGate to a sequence of keypoints.
    Returns:
        accepted_kp: (N, 3, 3) same as input, but with outliers replaced by np.nan
        rejection_mask: (N, 3) boolean mask where True means outlier
    """
    N, J, _ = raw_kp.shape
    accepted_kp = raw_kp.copy()
    rejection_mask = np.zeros((N, J), dtype=bool)
    
    gates = [AnatomicalGate() for _ in range(J)]
    
    for i in range(N):
        for j in range(J):
            if gates[j].is_outlier(raw_kp[i, j], dt):
                rejection_mask[i, j] = True
                accepted_kp[i, j] = np.nan
                
    return accepted_kp, rejection_mask
