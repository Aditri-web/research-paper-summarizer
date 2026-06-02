import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(5, device=device)
    print(f"✓ Successfully created tensor on MPS: {x}")
else:
    print("⚠ MPS not available, will use CPU")
