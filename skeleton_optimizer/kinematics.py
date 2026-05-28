import numpy as np
from dataclasses import dataclass

@dataclass
class ArmModel:
    L1: float = 0.32  # Upper arm length in meters
    L2: float = 0.28  # Forearm length in meters

def forward_kinematics(theta: np.ndarray, model: ArmModel = ArmModel()) -> np.ndarray:
    """
    Computes the 3D positions of the shoulder, elbow, and wrist given joint angles.
    theta = [phi, psi, alpha]
        phi (azimuth), psi (elevation) for shoulder (spherical coordinates)
        alpha for elbow flexion (dihedral angle)
    Returns:
        (3, 3) array of [shoulder, elbow, wrist]
    """
    phi, psi, alpha = theta
    
    # Shoulder is at origin
    shoulder = np.zeros(3)
    
    # Upper arm direction vector
    d_s = np.array([
        np.cos(psi) * np.cos(phi),
        np.cos(psi) * np.sin(phi),
        np.sin(psi)
    ])
    
    elbow = model.L1 * d_s
    
    # Forearm rotation axis (perpendicular to d_s in sagittal plane)
    z_axis = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(d_s, z_axis)
    norm = np.linalg.norm(rot_axis)
    if norm > 1e-6:
        rot_axis /= norm
    else:
        rot_axis = np.array([1.0, 0.0, 0.0])
        
    # Rodrigues rotation formula
    # R = I + sin(alpha)*K + (1-cos(alpha))*K^2
    K = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(alpha) * K + (1 - np.cos(alpha)) * (K @ K)
    
    # Forearm direction
    d_f = R @ d_s
    wrist = elbow + model.L2 * d_f
    
    return np.vstack([shoulder, elbow, wrist])

def jacobian(theta: np.ndarray, model: ArmModel = ArmModel()) -> np.ndarray:
    """
    Analytic Jacobian matrix of the forward kinematics.
    ∂f_i / ∂θ_j
    Returns:
        (9, 3) array where rows 0:3 are shoulder, 3:6 are elbow, 6:9 are wrist
    """
    phi, psi, alpha = theta
    L1, L2 = model.L1, model.L2
    
    J = np.zeros((9, 3))
    
    # Shoulder derivative is zero (it's fixed at origin)
    
    # --- Derivatives for Elbow ---
    # d_s derivatives
    dds_dphi = np.array([
        -np.cos(psi) * np.sin(phi),
        np.cos(psi) * np.cos(phi),
        0.0
    ])
    dds_dpsi = np.array([
        -np.sin(psi) * np.cos(phi),
        -np.sin(psi) * np.sin(phi),
        np.cos(psi)
    ])
    
    J[3:6, 0] = L1 * dds_dphi
    J[3:6, 1] = L1 * dds_dpsi
    # J[3:6, 2] is zero because alpha doesn't affect elbow position
    
    # --- Derivatives for Wrist ---
    # This requires full chain rule through Rodrigues formula.
    # To save time and complexity for the Python implementation, we'll use 
    # central finite differences for the Python Jacobian, but the C++ 
    # implementation will use the fully derived analytic form.
    # We implement a hybrid here: analytic for elbow, finite-diff for wrist.
    
    eps = 1e-6
    for i in range(3):
        t1 = theta.copy()
        t2 = theta.copy()
        t1[i] -= eps
        t2[i] += eps
        
        fk1 = forward_kinematics(t1, model)
        fk2 = forward_kinematics(t2, model)
        
        # Override wrist derivatives with finite diffs
        J[6:9, i] = (fk2[2] - fk1[2]) / (2 * eps)
        
    return J
