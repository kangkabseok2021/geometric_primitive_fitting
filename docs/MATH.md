# Skeletal Kinematics Mathematics

## 1. Forward Kinematics
The state is parameterized by $\theta = [\phi, \psi, \alpha]^T$:
- $\phi$: Shoulder azimuth
- $\psi$: Shoulder elevation
- $\alpha$: Elbow flexion

The shoulder is fixed at the origin $\mathbf{s} = [0, 0, 0]^T$.

Upper arm direction $\mathbf{d}_s$:
$$ \mathbf{d}_s = \begin{bmatrix} \cos(\psi)\cos(\phi) \\ \cos(\psi)\sin(\phi) \\ \sin(\psi) \end{bmatrix} $$

Elbow position $\mathbf{e}$:
$$ \mathbf{e} = L_1 \mathbf{d}_s $$

The forearm direction $\mathbf{d}_f$ is obtained by rotating $\mathbf{d}_s$ around an axis $\mathbf{u}$ perpendicular to the sagittal plane by angle $\alpha$ using the Rodrigues rotation formula:
$$ \mathbf{u} = \frac{\mathbf{d}_s \times \mathbf{z}}{\|\mathbf{d}_s \times \mathbf{z}\|} $$
$$ \mathbf{R}(\mathbf{u}, \alpha) = \mathbf{I} + \sin(\alpha)[\mathbf{u}]_\times + (1-\cos(\alpha))[\mathbf{u}]_\times^2 $$
$$ \mathbf{d}_f = \mathbf{R} \mathbf{d}_s $$

Wrist position $\mathbf{w}$:
$$ \mathbf{w} = \mathbf{e} + L_2 \mathbf{d}_f $$

## 2. Unscented Kalman Filter
The state tracks $[x, y, z, v_x, v_y, v_z]^T$ for each joint. A constant velocity model is used. The posterior covariance $\mathbf{P}$ yields an uncertainty metric for each joint's position, proportional to $\det(\mathbf{P})$.

## 3. Levenberg-Marquardt Optimization
We seek to minimize the weighted residual between the kinematic model and the UKF-smoothed observations $\mathbf{y}_i$:
$$ E(\theta) = \sum_{i=1}^3 w_i \|\mathbf{y}_i - \mathbf{f}_i(\theta)\|^2 $$
Where $w_i = 1 / \sqrt{\det(\mathbf{P}_i)}$.

The LM update step:
$$ (\mathbf{J}^T \mathbf{W} \mathbf{J} + \lambda \text{diag}(\mathbf{J}^T \mathbf{W} \mathbf{J})) \Delta\theta = \mathbf{J}^T \mathbf{W} (\mathbf{y} - \mathbf{f}(\theta)) $$
Where $\mathbf{J}$ is the Jacobian $\partial \mathbf{f} / \partial \theta$.
