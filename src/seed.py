"""Fix random seeds across Python, NumPy and PyTorch for reproducible experiments."""

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed()
    print(f"Seed fixed to {DEFAULT_SEED}.")
    print(f"  python random: {random.random():.6f}")
    print(f"  numpy:         {np.random.rand():.6f}")
    print(f"  torch:         {torch.rand(1).item():.6f}")
