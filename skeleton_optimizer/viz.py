import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from .kinematics import ArmModel, forward_kinematics

def plot_skeleton_3d(observed: np.ndarray, optimized_theta: np.ndarray, model: ArmModel):
    """
    Renders a single frame in matplotlib 3D.
    """
    fk = forward_kinematics(optimized_theta, model)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot observed (noisy) points
    ax.scatter(observed[:, 0], observed[:, 1], observed[:, 2], c='r', marker='x', label='Observed (Noisy)', s=100)
    
    # Plot optimized skeleton
    ax.plot(fk[:, 0], fk[:, 1], fk[:, 2], 'b-o', linewidth=3, markersize=8, label='Optimized Skeleton')
    
    # Constraints
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Constrained Skeletal Kinematics Fit')
    
    return fig

def create_dashboard(theta_seq: np.ndarray, residuals: np.ndarray, output_html: str = "swing_analysis.html"):
    """
    Creates an interactive Plotly dashboard for the swing metrics.
    User preferred 'dark mode'.
    """
    N = len(theta_seq)
    frames = np.arange(N)
    
    fig = go.Figure()
    
    # Add joint angles
    fig.add_trace(go.Scatter(x=frames, y=np.degrees(theta_seq[:, 0]), mode='lines', name='Shoulder Azimuth (deg)'))
    fig.add_trace(go.Scatter(x=frames, y=np.degrees(theta_seq[:, 1]), mode='lines', name='Shoulder Elevation (deg)'))
    fig.add_trace(go.Scatter(x=frames, y=np.degrees(theta_seq[:, 2]), mode='lines', name='Elbow Flexion (deg)'))
    
    # Add residual secondary axis
    fig.add_trace(go.Scatter(x=frames, y=residuals, mode='lines', name='LM Residual', 
                             line=dict(color='rgba(255, 0, 0, 0.5)', dash='dot'),
                             yaxis='y2'))
                             
    fig.update_layout(
        title='Skeletal Kinematics Optimizer - Joint Trajectories',
        xaxis_title='Frame',
        yaxis_title='Angle (Degrees)',
        yaxis2=dict(
            title='Optimization Residual',
            overlaying='y',
            side='right'
        ),
        template='plotly_dark', # Dark mode as requested
        hovermode='x unified'
    )
    
    fig.write_html(output_html)
    return fig
