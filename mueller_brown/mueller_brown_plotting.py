"""
Visualizations for the Mueller-Brown system. Torch tensors are converted to numpy only here.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from mueller_brown_analysis import (evaluate_on_grid, noised_energy_landscape, vp_coefficients, boltzmann_probabilities,
                                    grid_to_points, noise_samples)
from mueller_brown_potential import mueller_brown_potential, MB_MINIMA, MB_SADDLES, MB_STATE_NAMES


#############################################
# shared helpers
#############################################

def add_potential_contours(ax, X, Y, levels=15, u_max=100.0):
    """Draws the contour lines of the Mueller-Brown potential (clipped at u_max) as reference."""
    u = evaluate_on_grid(mueller_brown_potential, X, Y).clamp(max=u_max)
    style = dict(colors="k", linewidths=0.5, alpha=0.4)
    ax.contour(X.numpy(), Y.numpy(), u.numpy(), levels=levels, **style)


def add_info_box(ax, lines):
    """Text box in the upper right corner of the axes."""
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))


def add_stationary_points(ax, minima=MB_MINIMA, saddles=MB_SADDLES):
    """Marks the minima and saddle points of the potential."""
    minima, saddles = minima.numpy(), saddles.numpy()
    ax.scatter(minima[:, 0], minima[:, 1], marker="x", s=60, color="red", label="minima")
    ax.scatter(saddles[:, 0], saddles[:, 1], marker="x", s=60, color="black", edgecolors="black", label="saddle points")


#############################################
# potential and Boltzmann density
#############################################

def visualize_mueller_brown_potential(X, Y, n_contour_levels=15, u_max=100.0):
    """
    Visualizes the Mueller-Brown potential energy surface U(x, y) as a filled contour plot with contour
    lines. Values above u_max are clipped so that the contour levels resolve the low-energy region.

    Parameters
    ----------
    X, Y : torch.tensor
        meshgrid tensors on which the potential is evaluated
    n_contour_levels : int
        number of contour levels
    u_max : float
        upper limit of the displayed potential values
    """
    u = evaluate_on_grid(mueller_brown_potential, X, Y).clamp(max=u_max).numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    filled = ax.contourf(X.numpy(), Y.numpy(), u, levels=n_contour_levels, cmap="viridis")
    ax.contour(X.numpy(), Y.numpy(), u, levels=n_contour_levels, colors="k", linewidths=0.4, alpha=0.6)
    fig.colorbar(filled, ax=ax, label="U(x, y)")

    add_stationary_points(ax)
    ax.legend(loc="lower left", framealpha=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Müller-Brown potential")
    plt.show()


def visualize_boltzmann_density(X, Y, kT, n_contour_levels=15):
    """
    Visualizes the Boltzmann density p(x) ∝ exp(-U(x) / k_B T) of the Mueller-Brown potential as a filled
    contour plot.

    Parameters
    ----------
    X, Y : torch.tensor
        meshgrid tensors of a regular grid on which the density is evaluated
    kT : float
        thermal energy k_B T
    n_contour_levels : int
        number of contour levels
    """
    # cell_area = ((X[1, 0] - X[0, 0]) * (Y[0, 1] - Y[0, 0])).item()
    # density = boltzmann_probabilities(grid_to_points(X, Y), kT).reshape(X.shape) / cell_area

    points = grid_to_points(X, Y)
    density = boltzmann_probabilities(points, kT).reshape(X.shape)
    density = density.detach().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    filled = ax.contourf(X.numpy(), Y.numpy(), density, levels=n_contour_levels, cmap="viridis")
    ax.contour(X.numpy(), Y.numpy(), density, levels=n_contour_levels, colors="k", linewidths=0.4, alpha=0.6)
    fig.colorbar(filled, ax=ax, label="density")
    add_info_box(ax, [rf"$k_B T = {kT:g}$"])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Boltzmann density $\propto \exp(-U(x, y) / k_B T)$")
    plt.show()


#############################################
# Langevin samples
#############################################


def visualize_samples_density(samples, start_point, X, Y, config, v0, bins=100, x_range=(-1.5, 1.2), y_range=(-0.5, 2.0), n_contour_levels=15):
    """
    Visualizes the density of the positions of a Langevin trajectory as 2D histogram, with the contour lines of the
    Mueller-Brown potential as reference and the start point marked.

    Parameters
    ----------
    samples : torch.tensor
        positions of the trajectory of shape (n_steps + 1, 2), samples[0] is the start point
    X, Y : torch.tensor
        meshgrid tensors for the contour lines of the potential
    config : LangevinConfig
        simulation parameters shown in the info box
    v0 : torch.tensor
        start velocity of shape (1, 2)
    bins : int
        number of histogram bins per axis
    x_range, y_range : tuple
        histogram and plotting range
    n_contour_levels : int
        number of contour levels of the potential
    """
    data = samples.detach().numpy()
    start = start_point.reshape(-1).tolist()
    v0 = v0.reshape(-1).tolist()

    fig, ax = plt.subplots(figsize=(7, 6))
    *_, img = ax.hist2d(data[:, 0], data[:, 1], bins=bins, range=[x_range, y_range], density=True, cmap="turbo_r", rasterized=True, norm=LogNorm())
    fig.colorbar(img, ax=ax, label="density (log scale)")
    add_potential_contours(ax, X, Y, levels=n_contour_levels)

    ax.scatter(*start, marker="*", s=250, color="white", edgecolors="black", zorder=5, label=f"start ({start[0]:.2f}, {start[1]:.2f})")
    ax.legend(loc="lower left", framealpha=0.8)
    add_info_box(ax, config.info_lines() + [rf"$v_0 = ({v0[0]:.2f},\ {v0[1]:.2f})$"])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Density of Langevin integration samples (N = {len(data):,})")
    plt.show()


def visualize_running_state_fractions(states, fractions, dt, expected_state_fractions, names=MB_STATE_NAMES, first_step=0):
    """
    Plots the running state fractions over the simulation time (top) and the state trace (bottom).

    Parameters
    ----------
    states : torch.tensor
        state index of each sample, shape (n_samples,)
    fractions : torch.tensor
        running state fractions of shape (n_samples, n_states), see analysis.running_state_fractions
    dt : float
        integration step size, used for the time axis
    expected_state_fractions : torch.tensor
        expected state fractions of shape (n_states,), e.g. from analysis.boltzmann_state_fractions
    names : list of str
        names of the states
    first_step : int
        step index of the first sample, i.e. the burn-in length, so the x-axis shows the actual simulation steps
    """
    time = first_step + np.arange(len(states))
    fig, ax = plt.subplots(figsize=(9, 4))

    for k, name in enumerate(names):
        ax.plot(time, fractions[:, k].numpy(), color=f"C{k}", label=name)
        ax.axhline(expected_state_fractions[k].item(), color=f"C{k}", ls="--")

    ax.set_xlabel("simulation steps")
    ax.set_ylabel("running state fraction")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", title="dashed: Boltzmann")
    plt.tight_layout()
    plt.show()


#############################################
# noised energy landscapes
#############################################

def visualize_noised_energy_landscapes(samples, X, Y, vp, timesteps, seed, bins=80, x_range=(-1.5, 1.2), y_range=(-0.5, 2.0)):
    """
    Visualizes the energy landscapes of the noised Langevin samples for several diffusion times.

    The samples x_0 are noised with the VP-process,
        x_t = alpha_t * x_0 + sigma_t * eps,    eps ~ N(0, I),
    with alpha_t = exp(-1/2 int_0^t beta(s) ds) and sigma_t = sqrt(1 - alpha_t^2), always starting from x_0.
     The contour lines of the original potential are shown as reference.

    Parameters
    ----------
    samples : torch.tensor
        Langevin samples x_0 of shape (n_samples, 2)
    X, Y : torch.tensor
        meshgrid tensors for the contour lines of the original potential
    vp : VP
        variance preserving SDE providing log_mean_coeff(t) and std(t)
    timesteps : torch.tensor
        diffusion times t in [0, 1]
    seed : int
        seed of the noise generator, for reproducible figures
    bins : int
        number of histogram bins per axis
    x_range, y_range : tuple
        histogram and plotting range
    """
    alphas, sigmas = vp_coefficients(vp, timesteps)
    generator = torch.Generator().manual_seed(seed)

    n_cols = 4
    n_rows = int(np.ceil(len(timesteps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, t, alpha, sigma in zip(axes, timesteps.tolist(), alphas.tolist(), sigmas.tolist()):
        xt = noise_samples(samples.detach(), alpha, sigma, generator).numpy()
        *_, img = ax.hist2d(xt[:, 0], xt[:, 1], bins=bins, range=[x_range, y_range], density=True, norm=LogNorm(), cmap="turbo_r", rasterized=True)
        add_potential_contours(ax, X, Y, levels=15)
        fig.colorbar(img, ax=ax, label=r"$p_t$ (log scale)", shrink=0.85)

        ax.set_title(rf"$t = {t:g}$ | $\sigma_t = {sigma:.2f}$ | $\alpha_t = {alpha:.3f}$", fontsize=10)
        ax.set_aspect("equal")

    for ax in axes[len(timesteps):]:  # hide unused panels
        ax.set_visible(False)
    for ax in axes[::n_cols]:
        ax.set_ylabel(r"$y$")
    for ax in axes[(n_rows - 1) * n_cols:]:
        ax.set_xlabel(r"$x$")

    plt.tight_layout()
    plt.show()