import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass
from .kinematics import ArmModel, forward_kinematics, jacobian

@dataclass
class OptResult:
    theta_opt: np.ndarray
    residual: float
    n_iter: int
    converged: bool

def build_residuals(theta: np.ndarray, observed_points: np.ndarray, weights: np.ndarray, model: ArmModel) -> np.ndarray:
    """
    Weighted residuals for Levenberg-Marquardt.
    observed_points: (3, 3)
    weights: (3,) confidence weights per joint
    """
    fk = forward_kinematics(theta, model)
    r = np.zeros(9)
    for i in range(3):
        # w_i * (p_i - f_i(theta))
        r[3*i:3*i+3] = np.sqrt(weights[i]) * (observed_points[i] - fk[i])
    return r

def build_jacobian(theta: np.ndarray, observed_points: np.ndarray, weights: np.ndarray, model: ArmModel) -> np.ndarray:
    """
    Weighted Jacobian.
    """
    J = jacobian(theta, model)
    J_weighted = np.zeros_like(J)
    for i in range(3):
        J_weighted[3*i:3*i+3, :] = -np.sqrt(weights[i]) * J[3*i:3*i+3, :]
    return J_weighted

def solve_frame_py(observed: np.ndarray, weights: np.ndarray, model: ArmModel, theta_init: np.ndarray = None) -> OptResult:
    """
    Solves for theta using SciPy's Levenberg-Marquardt solver.
    """
    if theta_init is None:
        # Default starting guess
        theta_init = np.array([0.0, np.pi/4, np.pi/2])
        
    res = least_squares(
        build_residuals,
        x0=theta_init,
        jac=build_jacobian,
        method='lm',
        args=(observed, weights, model),
        ftol=1e-10,
        xtol=1e-10,
        max_nfev=500
    )
    
    return OptResult(
        theta_opt=res.x,
        residual=res.cost,
        n_iter=res.nfev,
        converged=res.success
    )

def solve_sequence_py(keypoints: np.ndarray, weights_seq: np.ndarray, model: ArmModel) -> list[OptResult]:
    """
    Solve across all frames with warm-starting.
    """
    N = len(keypoints)
    results = []
    theta_init = None
    
    for i in range(N):
        res = solve_frame_py(keypoints[i], weights_seq[i], model, theta_init)
        theta_init = res.theta_opt
        results.append(res)
        
    return results
