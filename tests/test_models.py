"""
Unit tests for GAN model architectures and training loop.
"""

import unittest

try:
    import torch
    from src.models import Generator, Discriminator
    from src.train import train_gan
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestModelsModule(unittest.TestCase):

    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in the active environment.")

    def test_generator_forward_pass(self):
        latent_dim = 16
        hidden_dim = 32
        output_dim = 50
        batch_size = 8

        g = Generator(latent_dim, hidden_dim, output_dim)
        z = torch.randn(batch_size, latent_dim)
        out = g(z)

        self.assertEqual(out.shape, (batch_size, output_dim))

    def test_discriminator_forward_pass(self):
        input_dim = 50
        hidden_dim = 32
        batch_size = 8

        d = Discriminator(input_dim, hidden_dim)
        x = torch.randn(batch_size, input_dim)
        out = d(x)

        self.assertEqual(out.shape, (batch_size, 1))
        # Sigmoid output must be in (0, 1)
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_train_gan_single_epoch(self):
        input_dim = 20
        n_samples = 40
        data = torch.randn(n_samples, input_dim)

        g_loss, d_loss, gen = train_gan(
            tensor_data=data,
            latent_dim=8,
            hidden_dim=16,
            lr=0.001,
            epochs=2,
            random_seed=42
        )

        self.assertIsInstance(g_loss, float)
        self.assertIsInstance(d_loss, float)
        self.assertIsInstance(gen, Generator)


if __name__ == "__main__":
    unittest.main()
