import torch
from torch import nn
from tqdm import tqdm
from .VP import VP

class Model(nn.Module):
    """Energy-based MLP; score is -grad_x E_theta (x, t)."""

    def __init__(self, dim=2, hidden_dim=92, n_layers=3):
        super().__init__()
        self.dim = dim

        layers = []
        in_dim = dim + 1  # data dims + noise level
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def energy(self, x: torch.Tensor, t: torch.Tensor):
        """Return the energy E_theta(x, t) with shape (N,)."""
        h = torch.cat([x, t.reshape(-1, 1)], dim=-1)  # (N, dim + 1)
        h = self.mlp(h)  # (N, 1)
        return h.squeeze(-1)  # (N,)

    def forward(self, x: torch.Tensor, t: torch.Tensor, create_graph: bool = True):
        """Return the score s_theta(x, t) = -grad_x E_theta (x, t) with shape (N, dim)."""
        x.requires_grad_(True)
        e = self.energy(x, t)
        # we only need create_graph if we want the gradient of the score (i.e. during training)
        return -torch.autograd.grad(e.sum(), x, create_graph=create_graph)[0]


    @torch.no_grad()
    def sample(self, model, num_samples, num_steps, seed=42, eps=1e-5):
        # TODO set seed
        model.eval()
        vp = VP()
        ts = torch.linspace(1, eps, num_steps + 1)
        x = torch.randn(num_samples, 2)

        for i in tqdm(range(num_steps), desc="Sampling"):
            t = torch.full((num_samples,), ts[i].item())
            dt = ts[i] - ts[i + 1]
            # enable gradient computation to compute the energy
            with torch.enable_grad():
                score = model(x, t, create_graph=False)
            drift, diffusion = vp.reverse_sde(x, t, score)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None] * torch.sqrt(dt) * torch.randn_like(x)
        return x