# A simple MLP with one hidden layer and tanh activation.
# Implemented using PyTorch library.

import torch.nn as nn
import torch.nn.functional as F


class MLP_torch(nn.Module):
    def __init__(self, n_inputs=1):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        """
        Args:
          x of shape (n_samples, n_inputs): Model inputs.

        Returns:
          y of shape (n_samples, 1): Model outputs.
        """
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x
