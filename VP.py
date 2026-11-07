import torch


def get_linear_beta_function(beta_min: float, beta_max: float):
    """
    Returns:
        beta(t): Linear beta schedule as a function of time.
        log_mean_coeff(t): Logarithm of the mean coefficient in the VP SDE.
    """

    def beta(t):
        return beta_min + t * (beta_max - beta_min)

    def log_mean_coeff(t):
        return -0.5 * t * beta_min - 0.25 * t**2 * (beta_max - beta_min)

    return beta, log_mean_coeff


class VP:
    """Variance preserving (VP) SDE, a.k.a. time rescaled Ohrnstein Uhlenbeck (OU) SDE."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta, self.log_mean_coeff = get_linear_beta_function(
            beta_min=beta_min,
            beta_max=beta_max
        )

    def sde(self, x, t):
        beta_t = self.beta(t)
        diffusion = torch.sqrt(beta_t)
        drift = -0.5 * beta_t[:, None] * x
        return drift, diffusion

    def std(self, t):
        return torch.sqrt(self.variance(t))

    def variance(self, t):
        return 1.0 - torch.exp(2 * self.log_mean_coeff(t))

    def marginal_prob(self, x, t):
        """Mean and std of the marginal at time t."""
        mean_coeff = torch.exp(self.log_mean_coeff(t))
        return mean_coeff * x, self.std(t)

    def reverse_sde(self, x, t, score):
        """Compute the reverse-time drift and diffusion from a score tensor."""
        drift, diffusion = self.sde(x, t)
        drift = -drift + diffusion[:, None] ** 2 * score
        return drift, diffusion
