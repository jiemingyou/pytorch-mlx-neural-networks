# A simple MLP with one hidden layer and tanh activation.
# Implemented using MLX library.

import mlx.core as mx
import mlx.nn as nn


class MLP_mlx(nn.Module):
    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 3)
        self.fc2 = nn.Linear(3, 1)

    def __call__(self, x):
        """
        Args:
          x of shape (n_samples, n_inputs): Model inputs.

        Returns:
          y of shape (n_samples, 1): Model outputs.
        """
        x = self.fc1(x)
        x = mx.tanh(x)
        x = self.fc2(x)
        return x
