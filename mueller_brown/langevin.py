"""
Underdamped Langevin dynamics with the BAOAB splitting scheme, plus saving and loading of trajectories.

Reference: B. Leimkuhler, C. Matthews, "Robust and efficient configurational molecular sampling via
Langevin dynamics" (2013).
"""
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass
class LangevinConfig:
    """Parameters of Langevin simulation."""
    dt: float = 0.005        # integration step size
    n_steps: int = 500_000        # number of integration steps
    kT: float = 20.0         # thermal energy k_B T
    gamma: float = 1.0       # collision rate
    mass: float = 1.0        # mass
    seed: int = 42                # seed for the randomly generated values (e.g. noise)

    def info_lines(self):
        """Lines for an info box in a plot."""
        return [rf"$\Delta t = {self.dt:g}$",
                rf"$n_{{\mathrm{{steps}}}} = {self.n_steps:,}$",
                rf"$k_B T = {self.kT:g}$",
                rf"$\gamma = {self.gamma:g}$",
                rf"$m = {self.mass:g}$"]


def langevin_integrator(x0, v0, force_fn, config):
    """
    BAOAB integration of underdamped Langevin dynamics.

    Parameters
    ----------
    x0 : torch.tensor
        start position of shape (1, 2)
    v0 : torch.tensor
        start velocity of shape (1, 2)
    force_fn : callable
        function that calculates the force based on the current position
    config : LangevinConfig
        simulation parameters

    Returns
    -------
    torch.tensor
        positions of the trajectory of shape (n_steps + 1, 2), the first row is the starting point
    """
    dt, kT, gamma, mass = config.dt, config.kT, config.gamma, config.mass

    points = [x0]
    simulation_progress_factor = 0.1

    # reparameterization to stick more to the actual param names in the paper
    p = v0 * mass  # momentum
    q = x0  # position

    for step in range(config.n_steps):

        if step == int(config.n_steps * simulation_progress_factor):
            print(f"Completed {int(simulation_progress_factor * 100)}% of the MD simulation")
            simulation_progress_factor += 0.1

        p = p + (dt / 2) * force_fn(q)                                                                         # B
        q = q + (dt / 2) * (1 / mass) * p                                                                      # A
        noise = torch.randn_like(q)
        p = math.exp(-gamma * dt) * p + (kT * (1 - math.exp(-2 * gamma * dt))) ** 0.5 * (mass ** 0.5) * noise  # O
        q = q + (dt / 2) * (1 / mass) * p                                                                      # A
        p = p + (dt / 2) * force_fn(q)                                                                         # B

        points.append(q)

    samples = torch.cat(points, dim=0)  # resulting to shape = [n_steps, 2]
    return samples


def save_trajectory(path, positions, config, x0, v0, burn_in=0):
    """
        Saves the positions of a trajectory as HDF5 file of shape (n_samples, 2). The first burn_in steps are discarded.
        The simulation parameters are stored as attributes of the file.
        """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as file:
        file.create_dataset("coordinates", data=positions[burn_in:].detach().numpy().astype(np.float32))
        for key, value in asdict(config).items():
            file.attrs[key] = value
        file.attrs["x0"] = x0.detach().numpy().reshape(-1)
        file.attrs["v0"] = v0.detach().numpy().reshape(-1)
        file.attrs["burn_in"] = burn_in


def load_trajectory(path):
    """
    Loads a trajectory saved with save_trajectory().

    Returns
    -------
    positions : torch.tensor
        positions of shape (n_samples, 2), burn-in already removed
    config : LangevinConfig
        simulation parameters
    x0, v0 : torch.tensor
        start position and start velocity of shape (1, 2)
    burn_in : int
        number of discarded initial steps
    """
    with h5py.File(path, "r") as file:
        positions = torch.from_numpy(file["coordinates"][:])
        attrs = dict(file.attrs)

    x0 = torch.from_numpy(attrs.pop("x0")).reshape(1, 2)
    v0 = torch.from_numpy(attrs.pop("v0")).reshape(1, 2)
    burn_in = int(attrs.pop("burn_in"))
    config = LangevinConfig(**{key: value.item() for key, value in attrs.items()})   # numpy -> python types
    return positions, config, x0, v0, burn_in