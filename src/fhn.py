"""
- Implements the vector field f(x)=[dv/dt, dw/dt] of the FitzHugh-Nagumo model
 and a differentiable RK4 for it.
- provides two simulators:
    - simulate_np: plain no-grad rollout for inspection
    - simulate_grad: tracks the trajectory and control with gradients for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint
from typing import Callable, Optional, Tuple
from torch import Tensor

class FHNVectorField(nn.Module):
    """
    Implements the vector field f(x)=[dv/dt, dw/dt] of the FitzHugh-Nagumo model.
    """

    def __init__(self, a: float = 0.7, b: float = 0.8, tau: float = 12.5):
        super(FHNVectorField, self).__init__()
        self.a = a
        self.b = b
        self.tau = tau

    def forward(self, t: Tensor, y: Tensor, I_ext: Optional[Tensor] = None) -> Tensor:
        """
        Compute the time derivative of the state.

        Args:
            t (Tensor): Current time (not used in this autonomous system).
            y (Tensor): Current state [v, w].
            I_ext (Optional[Tensor]): External input current.

        Returns:
            Tensor: Time derivative [dv/dt, dw/dt].
        """
        v, w = y[..., 0], y[..., 1]
        dvdt = v - (v ** 3) / 3 - w + (I_ext if I_ext is not None else 0)
        dwdt = (v + self.a - self.b * w) / self.tau
        return torch.stack([dvdt, dwdt], dim=-1)
    
    

def rk4_step(func: Callable, t: Tensor, y: Tensor, dt: float, I_ext: Optional[Tensor] = None) -> Tensor:
    """
    Perform a single RK4 step.

    Args:
        func (Callable): The vector field function.
        t (Tensor): Current time.
        y (Tensor): Current state.
        dt (float): Time step.
        I_ext (Optional[Tensor]): External input current.

    Returns:
        Tensor: Next state after time step dt.
    """
    k1 = func(t, y, I_ext)
    k2 = func(t + dt / 2, y + dt / 2 * k1, I_ext)
    k3 = func(t + dt / 2, y + dt / 2 * k2, I_ext)
    k4 = func(t + dt, y + dt * k3, I_ext)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate_np(
    func: Callable,
    y0: np.ndarray,
    t: np.ndarray,
    I_ext: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Simulate the FitzHugh-Nagumo model using a no-gradient RK4 integrator.

    Args:
        func (Callable): The vector field function.
        y0 (np.ndarray): Initial state [v0, w0].
        t (np.ndarray): Time points for simulation.
        I_ext (Optional[np.ndarray]): External input current.

    Returns:
        np.ndarray: Simulated states at each time point.
    """
    y = torch.tensor(y0, dtype=torch.float32)
    ys = [y.numpy()]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        I = torch.tensor(I_ext[i - 1], dtype=torch.float32) if I_ext is not None else None
        y = rk4_step(func, torch.tensor(t[i - 1], dtype=torch.float32), y, dt, I)
        ys.append(y.numpy())
    return np.array(ys)


def simulate_grad(
    func: Callable,
    y0: Tensor,
    t: Tensor,
    I_ext: Optional[Tensor] = None,
    method: str = 'dopri5'
) -> Tensor:
    """
    Simulate the FitzHugh-Nagumo model using a differentiable ODE solver.

    Args:
        func (Callable): The vector field function.
        y0 (Tensor): Initial state [v0, w0].
        t (Tensor): Time points for simulation.
        I_ext (Optional[Tensor]): External input current.
        method (str): ODE solver method.

    Returns:
        Tensor: Simulated states at each time point.
    """
    if I_ext is not None:
        # Wrap the function to include I_ext
        def func_with_input(t, y):
            idx = (t >= t_points).nonzero(as_tuple=True)[0][-1]
            return func(t, y, I_ext[idx])
        t_points = t.detach().cpu().numpy()
        ys = odeint(func_with_input, y0, t, method=method)
    else:
        ys = odeint(func, y0, t, method=method)
    return ys

# Example usage:
if __name__ == "__main__":
    fhn = FHNVectorField()
    y0 = np.array([0.0, 0.0])
    t = np.linspace(0, 100, 1000)
    I_ext = np.zeros_like(t)
    I_ext[100:200] = 0.5  # Example external input

    # No-gradient simulation
    traj_np = simulate_np(fhn.forward, y0, t, I_ext)

    # Gradient-tracking simulation
    y0_tensor = torch.tensor(y0, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)
    I_ext_tensor = torch.tensor(I_ext, dtype=torch.float32)
    traj_grad = simulate_grad(fhn.forward, y0_tensor, t_tensor, I_ext_tensor)