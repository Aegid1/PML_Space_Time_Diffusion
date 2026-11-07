"""
Analysis of Mueller-Brown trajectories: grids, Boltzmann distribution, histograms, metastable states,
comparison metrics, autocorrelation and the noising of samples with a VP forward process.
"""
import torch
import torch.nn.functional as F
from mueller_brown_potential import mueller_brown_potential, MB_MINIMA


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


def compare_to_boltzmann(samples, kT, bins=50, x_range=(-1.5, 1.2), y_range=(-0.5, 2.0)):
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


def boltzmann_state_fractions(kT, minima=MB_MINIMA, x_range=(-2.0, 1.5), y_range=(-1.0, 2.5), n_grid=400):
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