"""
Analysis of Mueller-Brown trajectories: grids, Boltzmann distribution, histograms, metastable states,
comparison metrics, autocorrelation and the noising of samples with a VP forward process.
"""
import torch
import torch.nn.functional as F
from mueller_brown_potential import mueller_brown_potential, MB_MINIMA, MB_STATE_NAMES


#############################################
# GRIDS
#############################################

def make_grid(x_range, y_range, n):
    """Regular meshgrid of shape (n, n)"""
    x = torch.linspace(*x_range, n)
    y = torch.linspace(*y_range, n)
    return torch.meshgrid(x, y)


def grid_to_points(X, Y):
    """Flattens meshgrid tensors into 2-dim tensor with points of shape (n_x * n_y, 2)."""
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def evaluate_on_grid(fn, X, Y):
    """Evaluates a function of points (N, 2) -> (N,) on a meshgrid and returns the values in grid shape."""
    return fn(grid_to_points(X, Y)).reshape(X.shape).detach()


#############################################
# BOLTZMANN DISTRIBUTION
#############################################

def boltzmann_probabilities(points, kT):
    """
    Calculates Boltzmann probabilities 1/Z * exp(-U(x, y) / k_B T) of Mueller-Brown potential.

    Returns
    -------
    torch.tensor
        tensor of shape (n_points,) that contains the boltzmann probabilities for all points of a grid
    """
    u = mueller_brown_potential(points)
    boltzmann_values = torch.exp(-u / kT)
    return boltzmann_values / torch.sum(boltzmann_values)


def compare_to_boltzmann_tv_kl(samples, kT, bins=50, x_range=(-1.5, 1.5), y_range=(-0.5, 2.0)):
    """
    Compares the empirical distribution of the simulation samples with the analytic
    Boltzmann distribution on the grid given by X, Y.

    Note: KL(langevin_samples_density || boltzmann_density) is allowed because boltzmann_density > 0 however the density
    of the simulation results can be zero at certain points for which KL(boltzmann_density || langevin_samples_density)
    would not be defined.

    Parameters
    ----------
    samples : torch.tensor
        positions of shape (n_samples, 2), burn-in already removed
    kT : float
        thermal energy of the simulation
    bins : int
        number of bins per axis
    x_range, y_range : tuple
        range of the comparison grid; samples outside are ignored
    """

    counts, (x_edges, y_edges) = torch.histogramdd(samples.detach(), bins=bins, range=[*x_range, *y_range])  # shape=[bins, bins]
    langevin_samples_density = counts / torch.sum(counts)

    # calculate the middle points of the bins, which are then used for the grid of the boltzmann density calculation
    x = 0.5 * (x_edges[1:] + x_edges[:-1])
    y = 0.5 * (y_edges[1:] + y_edges[:-1])
    X, Y = torch.meshgrid(x, y)  # shape=[bins, bins]
    coordinates = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    boltzmann_density = boltzmann_probabilities(coordinates, kT)
    boltzmann_density = boltzmann_density.reshape(X.shape) # transforming to grid for visualization

    # total variation distance
    tv_dist = 0.5 * torch.sum(torch.abs(langevin_samples_density - boltzmann_density))

    # KL-divergence
    mask = langevin_samples_density > 0
    kl = torch.sum(langevin_samples_density[mask] * (torch.log(langevin_samples_density[mask]) - torch.log(boltzmann_density[mask])))

    print(f"total variation distance: {tv_dist:.3f} | KL(P || Q): {kl:.3f} | bins: {bins}x{bins}")


def boltzmann_state_fractions(kT, minima=MB_MINIMA, x_range=(-1.5, 1.5), y_range=(-0.5, 2.0), n_grid=400):
    """
    Expected fractions of the metastable states under the Boltzmann distribution, obtained by numerical
    integration on a regular grid. Each grid point is assigned to its nearest minimum, i.e. with the same
    rule as the simulation samples. The grid should cover practically all probability mass.

    Returns
    -------
    torch.tensor
        tensor of shape (n_states,) that sums to 1
    """
    X, Y = make_grid(x_range, y_range, n_grid)
    points = grid_to_points(X, Y)

    weights = boltzmann_probabilities(points, kT)
    states = assign_states(points, minima)
    return torch.bincount(states, weights=weights)


#############################################
# METASTABLE STATES AND CONVERGENCE
#############################################

def assign_states(points, minima=MB_MINIMA):
    """
    Assigns each point of shape (N, 2) to the index of its nearest minimum out of the 3 minima that are stored in the
    torch tensor with shape (3, 2).
    """
    return torch.cdist(points.detach(), minima).argmin(dim=1)


def running_state_fractions(states, n_states):
    """
    Calculates the fraction of samples that lie in each state for every timestep n.

    Returns
    -------
    torch.tensor
        tensor of shape (n_samples, n_states)
    """

    n = torch.arange(1, len(states) + 1)

    # calculates the cumulative counts for every timestep
    state_counts = [(states == k).cumsum(dim=0) for k in range(n_states)]
    state_counts_tensor = torch.stack(state_counts, dim=1) # shape=[n_samples, n_states]
    state_fractions = state_counts_tensor / n[:, None]

    return state_fractions


#############################################
# NOISING WITH THE VP FORWARD PROCESS
#############################################

def vp_coefficients(vp, timesteps):
    """
    Coefficients for the VP noise process:
    alpha_t = exp(-1/2 int_0^t beta(s) ds) and sigma_t = sqrt(1 - alpha_t^2).
    """
    alphas = torch.exp(vp.log_mean_coeff(timesteps))
    sigmas = vp.std(timesteps)
    return alphas, sigmas


def noise_samples(samples, alpha, sigma, generator=None):
    """Noises the samples x_0 with: x_t = alpha * x_0 + sigma * eps, eps ~ N(0, I)."""
    eps = torch.randn(samples.shape, generator=generator)
    return alpha * samples + sigma * eps


def noised_energy_landscape(samples, alpha, sigma, bins, x_range, y_range, generator=None):
    """
    Empirical energy landscape U_t = -log p_t of the noised langevin samples (in units of k_B T).

    Parameters
    ----------
    samples : torch.tensor
        original samples of shape (n_samples, 2)
    alpha, sigma : float
        coefficients of the perturbation kernel
    bins : int
        number of bins per axis
    x_range, y_range : tuple
        histogram range
    generator : torch.Generator, optional
        generator for the noise

    Returns
    -------
    U_t : torch.tensor
        energy landscape of shape (bins, bins) with inf in empty bins
    x_edges, y_edges : torch.tensor
        bin edges of shape (bins + 1,)
    """
    noised_samples = noise_samples(samples.detach(), float(alpha), float(sigma), generator)
    counts, (x_edges, y_edges) = torch.histogramdd(noised_samples.detach(), bins=bins, range=[*x_range, *y_range])
    density = counts / torch.sum(counts)
    energy = -torch.log(density)   # empty bins -> inf
    return energy, x_edges, y_edges

#############################################
# FREE ENERGY PROFILES
#############################################

def free_energy_difference(samples, kT, bins, x_range, y_range, max_energy=10.0):
    """
    Difference between the relative free energy profile of samples and the true free energy profile U / k_B T on the
    grid X, Y, in units of k_B T.

    The free energy profile of the samples is estimated from a 2D histogram with one bin per grid point with
    F_samples = -log p. Both profiles are only defined up to a constant, which is removed by the mean difference
    over the compared bins. Compared are all bins whose true free energy lies less than max_energy above the
    minimum and that contain at least one sample, such that all other bins are set to nan.

    Parameters
    ----------
    samples : torch.tensor
        samples of shape (n_samples, 2)
    kT : float
        thermal energy k_B T
    max_energy : float
        upper limit of the true free energy (relative to its minimum) in units of k_B T for the comparison

    Returns
    -------
    difference : torch.tensor
        F_samples - F_true - offset of shape X.shape, nan outside the compared bins
    relevant : torch.tensor
        boolean mask of shape X.shape of all bins below max_energy, including empty ones
    """
    counts, (x_edges, y_edges) = torch.histogramdd(samples.detach(), bins=bins, range=[*x_range, *y_range])
    F_samples = -torch.log(counts / counts.sum())  # empty bins go to inf
    F_samples = F_samples - F_samples[torch.isfinite(F_samples)].min()  # setting the min at 0 for non-empty bins

    # Mueller-Brown values at the bin centers, so that both profiles lie on the same points
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    X, Y = torch.meshgrid(x_centers, y_centers)
    F_true = evaluate_on_grid(mueller_brown_potential, X, Y) / kT
    F_true = F_true - F_true.min() # setting the min to 0

    # mask to only compare bins with relevant energy range and at least one sample
    relevant = F_true < max_energy
    compared = relevant & torch.isfinite(F_samples)

    difference = F_samples - F_true
    difference[~compared] = float("nan")
    return difference, F_true, X, Y, relevant


def free_energy_difference_mae_per_state(difference, X, Y, minima=MB_MINIMA, names=MB_STATE_NAMES):
    """
    Mean absolute difference of two free energy profiles for each metastable state, in units of k_B T.

    Each bin is assigned to its nearest minimum, i.e. with the same rule as the samples for the state
    fractions. Bins without samples (nan in difference) are excluded.

    Parameters
    ----------
    difference : torch.tensor
        difference of the free energy profiles of shape X.shape, nan outside the compared bins
    X, Y : torch.tensor
        meshgrid tensors (indexing="ij") of the bin centers
    minima : torch.tensor
        positions of the minima of shape (n_states, 2)
    names : list of str
        names of the states

    Returns
    -------
    dict
        for each state: "mae" in units of k_B T and "n_bins" (number of compared bins)
    """
    states = assign_states(grid_to_points(X, Y), minima).reshape(X.shape)
    compared = ~torch.isnan(difference)

    results = {}
    for k, name in enumerate(names):
        valid = (states == k) & compared
        results[name] = difference[valid].abs().mean().item() if valid.any() else float("nan")

    return results