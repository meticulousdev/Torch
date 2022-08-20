import torch

print(f"mps.is_built(): {torch.backends.mps.is_built()}")
print(f"mps.is_available(): {torch.backends.mps.is_available()}")

# mps.is_built(): True
# mps.is_available(): True