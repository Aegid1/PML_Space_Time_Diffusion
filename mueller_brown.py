import math

import numpy as np
import torch
import matplotlib.pyplot as plt

MB_PARAMS = {
    "A" : torch.tensor([-200.0, -100.0, -170.0, 15.0]),   # amplitude
    "a" : torch.tensor([-1.0, -1.0, -6.5, 0.7]),    # width
    "b" : torch.tensor([0.0, 0.0, 11.0, 0.6]),      # orientation
    "c" : torch.tensor([-10.0, -10.0, -6.5, 0.7]),  #
    "x0" : torch.tensor([1.0, 0.0, -0.5, -1.0]),    #
    "y0" : torch.tensor([0.0, 0.5, 1.5, 1.0]),      #
}

def mueller_brown_potential(x):
    """
    Calculates the mueller-brown potential with the following formula:
    U(x, y) = Sum_i^4[ A_i * exp( a_i*(x - x0_i)^2 + b_i*(x - x0_i)*(y - y0_i) + c_i*(y - y0_i)^2 ) ]

    Parameters
    ----------
    x : torch.tensor
        Tensor of shape (N, 2), which contains (x, y)-coordinates

    Returns
    -------
    torch.tensor
        1D-tensor of shape (N,) with calculated potentials for each coordinate.
    """
    _, _, summands = _mueller_brown_terms(x, **MB_PARAMS)
    potential = torch.sum(summands, dim=1)
    return potential


def _mueller_brown_terms(x, A, a, b, c, x0, y0):
    """
    Helper function to calculate the summands of the mueller-brown potential formula and the differences of the provided
    coordinates (x,y) to the x0 and y0 points.

    Parameters
    ----------
    x : torch.tensor
        Tensor of shape (N, 2), which contains (x, y)-coordinates
    A, a, b, c, x0, y0 : torch.Tensor
        Tensor of shape (4,) containing the parameters of the mueller-brown potential function describing the surface.

    Returns
    -------
    dx : torch.Tensor
        Tensor of shape (N, 4) with the differences x - x0_i for every point and term.
    dy : torch.Tensor
        Tensor of shape (N, 4) with the differences y - y0_i for every point and term.
    summands : torch.Tensor
        Tensor of shape (N, 4) with the value of each of the four terms of the potential for every point.
        Summing over dim=1 calculates the potential U of shape (N,)
    """
    # splitting the coordinates in individual parts and calculating differences to x0, y0
    x_coord, y_coord = x[:, 0], x[:, 1]
    dx = x_coord[:, None] - x0[None, :]
    dy = y_coord[:, None] - y0[None, :]

    # calculation of relevant mueller_brown terms
    exponents = a[None, :] * (dx**2) + b[None, :] * dx * dy + c[None, :] * (dy**2)
    summands = A[None, :] * torch.exp(exponents)
    return dx, dy, summands


def mueller_brown_force(x):
    """
    Calculates the mueller-brown force F(x) = -\nabla U(x) where U(x) is the potential energy:
    """
    dx, dy, summands = _mueller_brown_terms(x, **MB_PARAMS)
    a, b, c = MB_PARAMS["a"], MB_PARAMS["b"], MB_PARAMS["c"]

    # calculating the forces in each direction and combining to final array
    dU_dx = -torch.sum(summands * (2 * a[None, :] * dx + b[None, :] * dy), dim=1)
    dU_dy = -torch.sum(summands * (2 * c[None, :] * dy + b[None, :] * dx), dim=1)
    return torch.stack((dU_dx, dU_dy), dim=1)


def langevin_integrator(x0, v0, force_fn, dt, n_steps, kT=1.0, gamma=1.0, mass=1.0):
    """
    BAOAB-Langevin integration from "Robust and efficient configurational molecular sampling via Langevin Dynamics" (2013).

    Parameters
    ----------
    x0 : torch.tensor
        starting position
    v0 : torch.tensor
        starting velocity
    force_fn :
        function that calculates the force based on the current position
    dt : float
        amount of change applied per step
    n_steps : int
        amount of integration steps
    kT : float
        scalar describing the thermal energy
    gamma : float
        scalar containing the collision rate parameter
    mass : float
        scalar mass of the atoms

    Returns
    -------
    torch.tensor
        2D-tensor of shape (n_steps + 1, 2) with calculated positions in each integration step
    """
    points = [x0]
    p = v0 * mass
    q = x0 # reparameterization to stick more to the actual param names in the paper

    for step in range(n_steps):
        force = force_fn(q)

        p = p + (dt/2) * force                                                                              # B
        q = q + (dt/2) * (1/mass) * p                                                                       # A
        noise = torch.randn_like(q)
        p = math.exp(-gamma * dt) * p + (kT * (1 - math.exp(-2 * gamma * dt)))**0.5 * (mass**0.5) * noise   # O
        q = q + (dt/2) * (1/mass) * p                                                                       # A
        p = p + (dt/2) * force                                                                              # B

        points.append(q)

    return torch.cat(points, dim=0)


def compare_mueller_brown_force_calculation(x):
    """
    Compares the analytical mueller-brown forces to the ones calculated with pytorch autograd
    """
    # analytical forces
    analytical_mb_force = mueller_brown_force(x)

    # autograd forces
    x.requires_grad = True
    mb_potential_tensor = mueller_brown_potential(x)
    torch.sum(mb_potential_tensor).backward() # calculate gradients to individual points

    calculation_differences = torch.abs(analytical_mb_force - (-x.grad))
    max_error = torch.max(calculation_differences)
    avg_error = torch.mean(calculation_differences)
    print(f"max absolute difference: {max_error} | avg absolute difference: {avg_error}")


def visualize_mueller_brown_potential(coordinates, n_contour_levels=15):
    """
    Creates a contour plot of the mueller-brown potential.

    parameters
    ----------
    coordinates: torch.tensor
        tensor containing the coordinates of the
    n_contour_levels : int
        number of contour levels
    """
    U = mueller_brown_potential(coordinates)
    U = U.reshape(X.shape).clamp(max=100.0) # transforming back to grid shape for visualization

    fig, ax = plt.subplots(figsize=(7, 6))

    # convert torch tensors back to numpy such that they work with matplotlib
    X_np, Y_np, U_np = X.numpy(), Y.numpy(), U.detach().numpy()

    filled = ax.contourf(X_np, Y_np, U_np, levels=n_contour_levels, cmap="viridis")
    ax.contour(X_np, Y_np, U_np, levels=n_contour_levels, colors="k", linewidths=0.4, alpha=0.6)
    fig.colorbar(filled, ax=ax, label="U(x, y)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("mueller-brown potential")
    ax.set_aspect("equal")
    plt.show()


def visualize_langevin_integration_samples_density(samples,
                                                   X,
                                                   Y,
                                                   bins,
                                                   x_range=(-1.5, 1.2),
                                                   y_range=(-0.5, 2.0)):
    """
    Creates a contour plot of the mueller-brown potential.

    parameters
    ----------
    coordinates: torch.tensor
        tensor containing the coordinates of the
    n_contour_levels : int
        number of contour levels
    """
    data = samples.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    _, _, _, img = ax.hist2d(data[:, 0], data[:, 1], bins=bins, range=[x_range, y_range], density=True, cmap="viridis")
    fig.colorbar(img, ax=ax, label="density")

    # visualizing the contour lines
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    U = mueller_brown_potential(coords)
    U = U.reshape(X.shape).clamp(max=100.0).numpy() # transforming back to grid shape for visualization
    ax.contour(X.numpy(), Y.numpy(), U, levels=15, colors="white", linewidths=0.6, alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Density of langevin integration samples (N = {len(data):,})")
    plt.show()


if __name__ == "__main__":
    # range of grid values
    x_range, y_range = (-1.5, 1.5), (-0.5, 2.0)
    n_axis_points = 300

    # calculation of grid
    x = torch.linspace(*x_range, n_axis_points)
    y = torch.linspace(*y_range, n_axis_points)
    X, Y = torch.meshgrid(x, y)
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    compare_mueller_brown_force_calculation(x=coords)
    visualize_mueller_brown_potential(coordinates=coords)

    # as the starting point a point close to one minimum of the potential surface is chosen
    start_point, start_velocity = torch.tensor([[1.0, 0.0]]), torch.randn(1,2)
    dt = 0.001
    n_steps = 100000
    force_fn = lambda x: mueller_brown_force(x)
    samples = langevin_integrator(x0=start_point, v0=start_velocity, force_fn=force_fn, dt=dt, n_steps=n_steps)
    visualize_langevin_integration_samples_density(samples, X=X, Y=Y, bins=100, x_range=x_range, y_range=y_range)