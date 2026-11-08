"""
Modular GAN training routines for pathway activity synthesis.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import Generator, Discriminator


def train_gan(
    tensor_data: torch.Tensor,
    latent_dim: int,
    hidden_dim: int,
    lr: float,
    epochs: int = 50,
    random_seed: int = 42
) -> Tuple[float, float, Generator]:
    """
    Train a GAN on pathway-level biomedical tensor data.

    Args:
        tensor_data: Standardized tensor of real pathway profiles, shape (N, input_dim).
        latent_dim: Dimensionality of latent noise vector z.
        hidden_dim: Number of hidden units in Generator and Discriminator.
        lr: Learning rate for Adam optimizers.
        epochs: Number of complete training passes over the dataset.
        random_seed: Random seed for initialization.

    Returns:
        g_loss: Final generator loss (float).
        d_loss: Final discriminator loss (float).
        generator: Trained Generator instance.
    """
    torch.manual_seed(random_seed)
    input_dim = tensor_data.size(1)

    generator = Generator(latent_dim, hidden_dim, input_dim)
    discriminator = Discriminator(input_dim, hidden_dim)

    loss_fn = nn.BCELoss()
    g_opt = optim.Adam(generator.parameters(), lr=lr)
    d_opt = optim.Adam(discriminator.parameters(), lr=lr)

    final_g_loss = 0.0
    final_d_loss = 0.0

    for _ in range(epochs):
        real = tensor_data
        batch_size = real.size(0)

        real_label = torch.ones(batch_size, 1)
        fake_label = torch.zeros(batch_size, 1)

        # 1. Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        z = torch.randn(batch_size, latent_dim)
        fake = generator(z)

        d_loss_real = loss_fn(discriminator(real), real_label)
        d_loss_fake = loss_fn(discriminator(fake.detach()), fake_label)
        d_loss = d_loss_real + d_loss_fake

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # 2. Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        z = torch.randn(batch_size, latent_dim)
        fake = generator(z)

        g_loss = loss_fn(discriminator(fake), real_label)

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        final_g_loss = float(g_loss.item())
        final_d_loss = float(d_loss.item())

    return final_g_loss, final_d_loss, generator
