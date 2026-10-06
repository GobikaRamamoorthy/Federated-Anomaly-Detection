import torch

# Load the global model
global_model = torch.load("federated_autoencoder_warmstart.pth", map_location="cpu")

# Broadcast (save copies for each substation)
for i in range(5):
    torch.save(global_model, f"substation_{i}_global.pth")
    print(f"✅ Saved global model for Substation {i} -> substation_{i}_global.pth")

print("🎯 All substations have received the global model successfully.")
