"""
Visualizations for the Mueller-Brown system. Torch tensors are converted to numpy only here.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, ListedColormap
from model.VP import VP
from mueller_brown_analysis import (evaluate_on_grid, noised_energy_landscape, vp_coefficients, boltzmann_probabilities,
                                    grid_to_points, noise_samples, assign_states)
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
# Mueller-Brown potential and Boltzmann density
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


def visualize_samples_density(samples, start_point, X, Y, config, v0, bins=100, x_range=(-1.5, 1.5), y_range=(-0.5, 2.0), n_contour_levels=15):
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

def visualize_noised_energy_landscapes_md(samples, X, Y, vp, timesteps, seed, bins=80, x_range=(-1.5, 1.5), y_range=(-0.5, 2.0)):
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

    n_cols = 3
    n_rows = int(np.ceil(len(timesteps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, t, alpha, sigma in zip(axes, timesteps.tolist(), alphas.tolist(), sigmas.tolist()):
        xt = noise_samples(samples.detach(), alpha, sigma, generator).numpy()

        # density of the noised samples -> energy U_t = -log p_t,
        density, x_edges, y_edges = np.histogram2d(xt[:, 0], xt[:, 1], bins=bins, range=[x_range, y_range],density=True)
        with np.errstate(divide="ignore"):
            energy = -np.log(density)  # empty bins -> inf
        energy = energy - energy[np.isfinite(energy)].min() # minimum shifted to 0
        energy[~np.isfinite(energy)] = np.nan  # empty bins stay white
        img = ax.pcolormesh(x_edges, y_edges, energy.T, vmin=0, cmap="turbo", rasterized=True)

        add_potential_contours(ax, X, Y, levels=15)

        ax.set_title(rf"$t = {t:g}$ | $\sigma_t = {sigma:.2f}$ | $\alpha_t = {alpha:.3f}$", fontsize=10)
        ax.set_aspect("equal")

    for ax in axes[len(timesteps):]:  # hide unused panels
        ax.set_visible(False)
    for ax in axes[::n_cols]:
        ax.set_ylabel(r"$y$")
    for ax in axes[(n_rows - 1) * n_cols:]:
        ax.set_xlabel(r"$x$")

    plt.tight_layout()
    fig.colorbar(img, ax=axes[:len(timesteps)].tolist(), label=r"relative energy [$k_BT$]", shrink=0.85, extend="max")
    plt.show()


def visualize_learned_energy_landscapes(model, X, Y, vp, timesteps, n_cols=3):
    """
    Visualizes the noised energy landscapes E_theta(x, t) learned by the diffusion model for several
    diffusion times.
    All energies are given in units of k_B T and shifted such that their minimum on the grid is 0, so that
    all panels share one color scale. The contour lines of the original potential are shown as reference.

    Parameters
    ----------
    model : torch.nn.Module
        trained diffusion model providing energy(x, t)
    X, Y : torch.tensor
        meshgrid tensors on which the energies are evaluated
    vp : VP
        variance preserving SDE providing log_mean_coeff(t) and std(t)
    timesteps : torch.tensor
        diffusion times t in [0, 1]
    n_cols : int
        number of panels per row
    """
    alphas, sigmas = vp_coefficients(vp, timesteps)
    points = grid_to_points(X, Y)

    # analytic energy at t = 0 as reference, then the learned energies for all diffusion times
    with torch.no_grad():
        energies = []
        for t in timesteps.tolist():
            times = torch.full((len(points), 1), t)
            energies.append(model.energy(points, times).reshape(-1))
    titles = [rf"$t = {t:g}$ | $\sigma_t = {s:.2f}$ | $\alpha_t = {a:.3f}$" for t, s, a in zip(timesteps.tolist(), sigmas.tolist(), alphas.tolist())]

    n_panels = len(energies)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, energy, title in zip(axes, energies, titles):
        energy = (energy - energy.min()).reshape(X.shape)          # minimum of each panel at 0
        img = ax.contourf(X.numpy(), Y.numpy(), energy.numpy(), levels=15, extend="max", cmap="turbo")
        add_potential_contours(ax, X, Y, levels=15)

        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

    for ax in axes[n_panels:]:  # hide unused panels
        ax.set_visible(False)
    for ax in axes[::n_cols]:
        ax.set_ylabel(r"$y$")
    for ax in axes[(n_rows - 1) * n_cols:]:
        ax.set_xlabel(r"$x$")

    plt.tight_layout()
    fig.colorbar(img, ax=axes[:n_panels].tolist(), label=r"relative energy [$k_BT$]", shrink=0.85)
    plt.show()


def visualize_generated_samples_density(model, X, Y, num_samples=50_000, num_steps=200, bins=100, x_range=(-1.5, 1.5), y_range=(-0.5, 2.0), save_path=None):
    """
    Generates samples with the trained diffusion model and visualizes their density as 2D histogram on a
    logarithmic color scale, with the contour lines of the Mueller-Brown potential as reference. Bins without samples stay white.

    Parameters
    ----------
    model : torch.nn.Module
        trained diffusion model providing the function sample()
    X, Y : torch.tensor
        meshgrid tensors for the contour lines of the original potential
    num_samples : int
        number of generated samples
    num_steps : int
        number of steps of the sampler
    bins : int
        number of histogram bins per axis
    x_range, y_range : tuple
        histogram and plotting range
    save_path : str or Path, optional
        if given, the generated samples are saved to this path as .npy file

    Returns
    -------
    torch.tensor
        generated samples of shape (num_samples, 2)
    """
    generated = model.sample(num_samples=num_samples, num_steps=num_steps).detach()
    if save_path is not None:
        np.save(save_path, generated.numpy())

    data = generated.numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    *_, img = ax.hist2d(data[:, 0], data[:, 1], bins=bins, range=[x_range, y_range], density=True, norm=LogNorm(), cmap="turbo_r", rasterized=True)
    fig.colorbar(img, ax=ax, label="density (log scale)")
    add_potential_contours(ax, X, Y, levels=15)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(f"Density of generated samples of the diffusion model (N = {num_samples:,})")
    plt.tight_layout()
    plt.show()
    return generated


#############################################
# free energy comparisons
#############################################

def visualize_free_energy_difference(difference, X, Y):
    """
    Plots the difference of two free energy profiles on the grid.
    Red: free energy too high (too few samples),
    blue: too low (too many samples)
    white: bins that are not compared
    """
    limit = np.nanmax(np.abs(difference.numpy())) # nanmax returns the max value with ignoring NaNs

    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.pcolormesh(X.numpy(), Y.numpy(), difference.numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
    fig.colorbar(img, ax=ax, label=r"$F_\mathrm{samples} - F_\mathrm{true}$ [$k_BT$]", shrink=0.8)

    ax.set_title("free energy profiles")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def visualize_free_energy_difference_per_state(difference, X, Y, results, minima=MB_MINIMA, names=MB_STATE_NAMES):
    """
    Plots the difference of two free energy profiles separately for each metastable state, in one row of
    subplots. In each subplot, the bins assigned to the state (nearest minimum) are colored by the difference,
    all other compared bins are shown in gray, and bins without samples stay white. The title shows the mean
    absolute difference of the state.

    Parameters
    ----------
    difference : torch.tensor
        difference of the free energy profiles of shape X.shape, nan outside the compared bins
    X, Y : torch.tensor
        meshgrid tensors (indexing="ij") of the bin centers
    results : dict
        output of free_energy_mae_per_state, containing the mae for each state
    minima : torch.tensor
        positions of the minima of shape (n_states, 2)
    names : list of str
        names of the states, same order as minima
    """
    limit = np.nanmax(np.abs(difference.numpy())) # nanmax returns the max value with ignoring NaNs

    states = assign_states(grid_to_points(X, Y), minima).reshape(X.shape).numpy()
    diff = difference.numpy()
    compared = ~np.isnan(diff)
    X_np, Y_np = X.numpy(), Y.numpy()
    gray = ListedColormap(["lightgray"])

    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.2))
    for k, (ax, name) in enumerate(zip(axes, names)):
        # define relevant and gray area of the subplot
        in_state = compared & (states == k)
        other = compared & (states != k)

        # compared bins of the other states in gray, bins of this state colored by the difference
        ax.pcolormesh(X_np, Y_np, np.where(other, 1.0, np.nan), cmap=gray, shading="nearest")
        img = ax.pcolormesh(X_np, Y_np, np.where(in_state, diff, np.nan), cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
        ax.scatter(minima[k, 0], minima[k, 1], marker="x", color="black", s=50, zorder=5) # plot the minima as an x

        mae = results[name]
        ax.set_title(f"{name}\nMAE = {mae:.2f} $k_BT$", fontsize=10)
        ax.set_xlabel(r"$x$")
        ax.set_aspect("equal")

    axes[0].set_ylabel(r"$y$")
    plt.tight_layout()
    fig.colorbar(img, ax=axes.tolist(), label=r"$F_\mathrm{samples} - F_\mathrm{true}$ [$k_BT$]", shrink=0.85, extend="both")
    plt.show()