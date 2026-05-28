import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

class JointUKF:
    """
    Unscented Kalman Filter for a single 3D joint.
    State x = [px, py, pz, vx, vy, vz]^T
    Measurement z = [px, py, pz]^T
    """
    def __init__(self, dt: float = 1.0/30.0, noise_sigma: float = 0.05):
        self.dt = dt
        
        # 6D state -> 13 sigma points
        self.points = MerweScaledSigmaPoints(n=6, alpha=1e-3, beta=2., kappa=0.)
        
        self.ukf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=dt, 
                                         fx=self.fx, hx=self.hx, 
                                         points=self.points)
        
        self.ukf.x = np.zeros(6)
        # Initial covariance
        self.ukf.P = np.eye(6) * 1.0
        
        # Process noise: small for position, larger for velocity (random walk)
        q_pos = np.eye(3) * 1e-4
        q_vel = np.eye(3) * 1e-2
        self.ukf.Q = block_diag(q_pos, q_vel)
        
        # Measurement noise: corresponds to observation noise
        self.ukf.R = np.eye(3) * (noise_sigma ** 2)

    @staticmethod
    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        """State transition function (constant velocity model)."""
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F @ x
        
    @staticmethod
    def hx(x: np.ndarray) -> np.ndarray:
        """Measurement function (observe position only)."""
        return x[:3]
        
    def reset(self, initial_pos: np.ndarray):
        """Reset the filter state."""
        self.ukf.x = np.zeros(6)
        self.ukf.x[:3] = initial_pos
        self.ukf.P = np.eye(6) * 1.0

    def predict_update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one step of predict and update.
        Returns:
            x_pos: (3,) smoothed position
            P_pos: (3,3) posterior covariance matrix of the position
        """
        self.ukf.predict()
        self.ukf.update(z)
        return self.ukf.x[:3].copy(), self.ukf.P[:3, :3].copy()

def filter_joint(observations: np.ndarray, dt: float = 1.0/30.0, noise_sigma: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply UKF to a sequence of observations for a single joint.
    Returns:
        smoothed_positions: (N, 3)
        posterior_covariances: (N, 3, 3)
    """
    N = len(observations)
    smoothed = np.zeros((N, 3))
    covs = np.zeros((N, 3, 3))
    
    ukf = JointUKF(dt, noise_sigma)
    if N > 0:
        ukf.reset(observations[0])
        
    for i in range(N):
        pos, cov = ukf.predict_update(observations[i])
        smoothed[i] = pos
        covs[i] = cov
        
    return smoothed, covs
