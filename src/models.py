"""
PyTorch Generative Adversarial Network (GAN) architectures
for biomedical pathway activity synthesis.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # fallback for type hints


if TORCH_AVAILABLE:
    class Generator(nn.Module):
        """
        Two-layer feed-forward Generator network mapping a low-dimensional
        latent noise vector to synthetic biomedical pathway activity vectors.
        """
        def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)


    class Discriminator(nn.Module):
        """
        Two-layer feed-forward Discriminator network classifying candidate
        pathway activity vectors as physical (1.0) or synthetic (0.0).
        """
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class Generator:
        def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
            raise ImportError("PyTorch is required for Generator. Install via `pip install torch`.")

    class Discriminator:
        def __init__(self, input_dim: int, hidden_dim: int):
            raise ImportError("PyTorch is required for Discriminator. Install via `pip install torch`.")
