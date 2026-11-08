"""
Mueller-Brown potential: parameters, potential energy, analytic force and its stationary points.

Reference: K. Müller, L. D. Brown, "Location of saddle points and minimum energy paths by a constrained
simplex optimization procedure" (1979).
"""

import torch

MB_PARAMS = {
    "A": torch.tensor([-200.0, -100.0, -170.0, 15.0]),   # amplitudes
    "a": torch.tensor([-1.0, -1.0, -6.5, 0.7]),          # widths in x
    "b": torch.tensor([0.0, 0.0, 11.0, 0.6]),            # orientations
    "c": torch.tensor([-10.0, -10.0, -6.5, 0.7]),        # widths in y
    "x0": torch.tensor([1.0, 0.0, -0.5, -1.0]),          # centers in x
    "y0": torch.tensor([0.0, 0.5, 1.5, 1.0]),            # centers in y
}

MB_MINIMA = torch.tensor([[-0.558, 1.442],   # A: upper left, global minimum
                          [0.623, 0.028],    # B: lower right
                          [-0.050, 0.467]])  # C: middle
MB_STATE_NAMES = ["A (upper left)", "B (lower right)", "C (middle)"]
MB_SADDLES = torch.tensor([[-0.822, 0.624],  # between A and C
                           [0.212, 0.293]])  # between C and B


def _mueller_brown_terms(x, A, a, b, c, x0, y0):
    """
    Helper function to calculate the summands of the Mueller-Brown potential and the differences of the
    coordinates to the centers x0_i, y0_i.

    Parameters
    ----------
    x : torch.tensor
        tensor of shape (N, 2) with (x, y)-coordinates
    A, a, b, c, x0, y0 : torch.tensor
        tensors of shape (4,) with the parameters of the potential

    Returns
    -------
    dx, dy : torch.tensor
        tensors of shape (N, 4) with the differences x - x0_i and y - y0_i
    summands : torch.tensor
        tensor of shape (N, 4) with the value of each of the four terms; summing over dim=1 gives U
    """
    dx = x[:, 0, None] - x0[None, :]
    dy = x[:, 1, None] - y0[None, :]

    exponents = a[None, :] * dx**2 + b[None, :] * dx * dy + c[None, :] * dy**2
    summands = A[None, :] * torch.exp(exponents)
    return dx, dy, summands


def mueller_brown_potential(x):
    """
    Calculates the Mueller-Brown potential
        U(x, y) = sum_i A_i * exp(a_i (x - x0_i)^2 + b_i (x - x0_i)(y - y0_i) + c_i (y - y0_i)^2).

    Parameters
    ----------
    x : torch.tensor
        tensor of shape (N, 2) with (x, y)-coordinates

    Returns
    -------
    torch.tensor
        tensor of shape (N,) with the potential energy of each point
    """
    _, _, summands = _mueller_brown_terms(x, **MB_PARAMS)
    return summands.sum(dim=1)


def mueller_brown_force(x):
    """
    Calculates the analytic Mueller-Brown force F(x) = -grad U(x).

    Parameters
    ----------
    x : torch.tensor
        tensor of shape (N, 2) with (x, y)-coordinates

    Returns
    -------
    torch.tensor
        tensor of shape (N, 2) with the force (F_x, F_y) at each point
    """
    dx, dy, summands = _mueller_brown_terms(x, **MB_PARAMS)
    a, b, c = MB_PARAMS["a"], MB_PARAMS["b"], MB_PARAMS["c"]

    F_x = -(summands * (2 * a[None, :] * dx + b[None, :] * dy)).sum(dim=1)
    F_y = -(summands * (2 * c[None, :] * dy + b[None, :] * dx)).sum(dim=1)
    return torch.stack((F_x, F_y), dim=1)


def compare_force_with_autograd(x, dtype=torch.float32):
    """
    Compares the analytic force with the negative gradient of the potential obtained via torch.autograd,
    with all computations performed in the given floating point precision.


    Parameters
    ----------
    x : torch.tensor
        tensor of shape (N, 2) with the test points
    dtype : torch.dtype
        floating point precision of the comparison, e.g. torch.float32 or torch.float64

    Returns
    -------
    tuple of float
        maximum and mean absolute deviation over all points and components
    """
    x = x.detach().to(dtype)
    analytic_force = mueller_brown_force(x)

    x_grad = x.clone().requires_grad_(True)
    mueller_brown_potential(x_grad).sum().backward()
    autograd_force = -x_grad.grad

    deviation = (analytic_force - autograd_force).abs()
    max_error, avg_error = deviation.max().item(), deviation.mean().item()
    print(f"{str(dtype):>13} | max absolute difference: {max_error:.3e} | avg absolute difference: {avg_error:.3e}")
    return max_error, avg_error