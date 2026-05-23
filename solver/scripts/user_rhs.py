"""Example Python right-hand-side functions for the C++ ODE solver engine."""
import numpy as np


def rhs(y, t):
    """Non-trivial scalar RHS: dy/dt = -0.5y + 0.1t²"""
    return -0.5 * y + 0.1 * (t ** 2)


def rhs_stiff(y, t):
    """Stiff RHS: dy/dt = -1000(y - t²) + 2t  (eigenvalue ≈ -1000)"""
    return -1000.0 * (y - t ** 2) + 2.0 * t


def batch_rhs(t_array):
    """Vectorised RHS for the entire time grid: f(t) = exp(-t).
    Accepts a list or NumPy array from the C++ caller."""
    return np.exp(-np.asarray(t_array))
