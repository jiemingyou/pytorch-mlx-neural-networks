# Data generation functions. For both MLX and PyTorch.

import torch
import mlx.core as mx


# MLP data generation functions
# -----------------------------


# f(x) = x * sin(2 * pi * x)
def noisy_sin_torch(n, device):
    x = torch.rand(n, 1).sort(dim=0)[0]
    y = x * torch.sin(2 * torch.pi * x)
    err = torch.normal(0, 0.1, size=(n, 1))
    y += err
    return x.to(device), y.to(device)


# f(x) = x * sin(2 * pi * x)
def noisy_sin_mlx(n):
    x = mx.random.uniform(low=0, high=1, shape=(n, 1))
    x = mx.sort(x, axis=0)
    y = x * mx.sin(2 * mx.pi * x)
    err = mx.random.normal(loc=0, scale=0.1, shape=(n, 1))
    y = y + err
    return x, y
