"""
Mueller-Brown experiments: Langevin simulation, comparison with the Boltzmann distribution, convergence
checks and noised energy landscapes.
"""
import os
import torch
from mueller_brown_analysis import make_grid, grid_to_points, compare_to_boltzmann, assign_states, running_state_fractions, boltzmann_state_fractions
from langevin import LangevinConfig, langevin_integrator, save_trajectory, load_trajectory
from mueller_brown_plotting import (visualize_mueller_brown_potential, visualize_boltzmann_density,
                                    visualize_samples_density, visualize_running_state_fractions,
                                    visualize_noised_energy_landscapes_md, visualize_generated_samples_density,
                                    visualize_learned_energy_landscapes)
from mueller_brown_potential import mueller_brown_force, compare_force_with_autograd, MB_MINIMA
from model.VP import VP
from model.Model import Model

TRAJECTORY_PATH = os.path.join("data", "mueller_brown", "traj0.h5")
BURN_IN = 10_000

x_range, y_range = (-1.5, 1.5), (-0.5, 2.0)
X, Y = make_grid(x_range, y_range, n=100)

config = LangevinConfig(dt=0.005, n_steps=500_000, kT=20.0, gamma=1.0, mass=1.0, seed=42)
start_point = torch.tensor([[1.0, 0.0]])       # near minimum B
# start_point = torch.tensor([[0.0, 0.5]])     # near minimum C
# start_point = torch.tensor([[-0.5, 1.5]])    # near minimum A


if __name__ == "__main__":
    vp = VP()
    #############################################
    # potential and force
    #############################################
    # compare_force_with_autograd(grid_to_points(X, Y), dtype=torch.float32)
    # compare_force_with_autograd(grid_to_points(X, Y), dtype=torch.float64)
    # visualize_mueller_brown_potential(X, Y)
    # visualize_boltzmann_density(X, Y, kT=config.kT)


    #############################################
    # langevin simulation
    #############################################

    # torch.manual_seed(config.seed)
    # start_velocity = torch.randn(1,2)
    # samples = langevin_integrator(start_point, start_velocity, mueller_brown_force, config)
    # save_trajectory(TRAJECTORY_PATH, samples, config, start_point, start_velocity, burn_in=BURN_IN)
    samples, config, start_point, start_velocity, burn_in = load_trajectory(TRAJECTORY_PATH)
    torch.manual_seed(config.seed)


    # visualize_samples_density(samples, samples[0].detach(), X, Y, config, start_velocity, bins=100, x_range=x_range, y_range=y_range)

    # comparison with the Boltzmann distribution
    # compare_to_boltzmann(samples, kT=config.kT, bins=50, x_range=x_range, y_range=y_range)

    # convergence: running state fractions vs. Boltzmann state fractions
    # states = assign_states(samples, MB_MINIMA)
    # fractions = running_state_fractions(states, n_states=len(MB_MINIMA))
    # expected_state_fraction = boltzmann_state_fractions(kT=config.kT)
    # visualize_running_state_fractions(states, fractions, dt=config.dt, expected_state_fractions=expected_state_fraction)


    # noised energy landscapes from molecular dynamics
    timesteps = torch.tensor([0.0, 0.03, 0.1, 0.17, 0.3, 1.0])
    visualize_noised_energy_landscapes_md(samples, X, Y, vp, timesteps, seed=config.seed, x_range=x_range, y_range=y_range)

    #############################################
    # learned model
    #############################################
    model = Model(dim=2, hidden_dim=92, n_layers=3)
    model.load_state_dict(torch.load(os.path.join("data", "mueller_brown", "model.pt")))
    X, Y = make_grid((-1.8, 1.1), (-0.5, 2.0), n=150)
    # generated = visualize_generated_samples_density(model, X, Y, num_samples=50_000, num_steps=200)

    timesteps = torch.tensor([0.0, 0.03, 0.1, 0.17, 0.3, 1.0])
    visualize_learned_energy_landscapes(model, X, Y, vp, timesteps)