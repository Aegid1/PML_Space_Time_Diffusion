import os
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.optim import AdamW
from model.Model import Model
from model.VP import VP

vp = VP()
SEED = 42
OUTPUT_DIR = os.path.join("..", "data", "mueller_brown")
DATA_DIR = os.path.join("..", "data")

def perturb(t, data):
    """
    Perturb the data x(0) using the forward SDE to get x(t).

    Returns:
        x_t: noisy sample
        noise: sampled noise
        std: standard deviation at time t
    """
    mean, std = vp.marginal_prob(data, t)
    noise = torch.randn_like(data)
    return mean + std * noise, noise, std


def errors(noise, score, std):
    return noise + std * score


def diffusion_loss_fn(batch, ts, model):
    x_t, noise, std = perturb(ts, batch)
    score = model(x_t, ts)
    e = errors(noise, score, std)
    return torch.mean(e**2)


def load_data(dataset="aldp", transformation=None, n_train=None, n_test=None, gap=0, root=DATA_DIR):
    """Return lazy training and test generators without shuffling the samples to avoid data leakage."""
    paths = {
        "aldp": "aldp/AImplicitUnconstrained/traj0.h5",
        "mueller_brown": "mueller_brown/traj0.h5"
    }
    path = Path(root) / paths[dataset]
    with h5py.File(path, "r") as data:
        n_samples = len(data["coordinates"])

    n_train = int(0.8 * (n_samples - gap)) if n_train is None else n_train
    n_test = n_samples - n_train - gap if n_test is None else n_test

    def generate(start, stop):
        with h5py.File(path, "r") as data:
            for i in range(start, stop):
                sample = torch.from_numpy(data["coordinates"][i].copy()).float()
                yield transformation(sample) if transformation is not None else sample

    return generate(0, n_train), generate(n_samples - n_test, n_samples)


def train_loop(data, loss_fn, epochs, BS, model, optimizer, validation_data):
    losses, validation_losses = [], []
    model.train()

    # We use fixed noise levels for validation
    validation_rng = torch.Generator().manual_seed(SEED + 1)
    validation_t = torch.rand(len(validation_data), 1, generator=validation_rng) * (1 - 1e-5) + 1e-5
    validation_noise = torch.randn(validation_data.shape, generator=validation_rng)
    mean, std = vp.marginal_prob(validation_data, validation_t)
    validation_x = mean + std * validation_noise

    def step(optimizer, model, data, idx):
        batch = data[idx]
        ts = torch.rand(BS, 1) * (1 - 1e-5) + 1e-5
        loss = loss_fn(batch, ts, model)
        loss.backward()
        optimizer.step()
        return loss

    for _ in tqdm(range(epochs)):
        perm = torch.randperm(data.size(0))[: data.size(0) // BS * BS]
        perm = perm.reshape(-1, BS)

        epoch_losses = []
        for idx in perm:
            optimizer.zero_grad()
            loss = step(optimizer, model, data, idx)
            epoch_losses.append(loss.item())

        losses.append(np.mean(epoch_losses))
        score = model(validation_x.detach(), validation_t, create_graph=False)
        valdation_loss = torch.mean(errors(validation_noise, score, std) ** 2)
        validation_losses.append(valdation_loss.item())

    return {"train": losses, "validation": validation_losses}


if __name__ == "__main__":
    model = Model(dim=2, hidden_dim=92, n_layers=3)
    optimizer = AdamW(model.parameters(), lr=3e-4)

    train_generator, test_generator = load_data("mueller_brown", gap=10_000)
    train_data, test_data = torch.stack(list(train_generator)), torch.stack(list(test_generator))
    print("Training samples:", len(train_data), "| Test samples:", len(test_data))

    train_history = train_loop(train_data, diffusion_loss_fn, 100, 512, model, optimizer, test_data)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    plt.figure(figsize=(5, 3))
    plt.title("Loss")
    plt.plot(train_history["train"], label="Training")
    plt.plot(train_history["validation"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.gcf().savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), bbox_inches="tight", dpi=180)
    plt.show()
    print("Final Loss:", {key: values[-1] for key, values in train_history.items()})
