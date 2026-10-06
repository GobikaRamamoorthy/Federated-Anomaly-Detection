import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# ========== Load New Data ==========
df_new = pd.read_csv("substation_0_new.csv")

# Keep only feature columns (drop label if exists)
if 'Label' in df_new.columns:
    X_new = df_new.drop(columns=['Label']).values
else:
    X_new = df_new.values

print(f"Loaded new data: {X_new.shape[0]} samples, {X_new.shape[1]} features")

# ========== Load Scaler ==========
with open('scaler_substation_3.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# ========== Define Autoencoder (same as training) ==========
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Initialize model
input_dim = X_new.shape[1]
model = Autoencoder(input_dim, latent_dim=16)

# Load trained model
model.load_state_dict(torch.load("substation_3_best_autoencoder.pth"))
model.eval()

# ========== Load Threshold ==========
with open('threshold_substation_3.pkl', 'rb') as f:
    threshold_info = pickle.load(f)
threshold = threshold_info['threshold']
print(f"Loaded threshold: {threshold:.6f}")

# ========== Compute Reconstruction Errors ==========
with torch.no_grad():
    reconstructed = model(X_new_tensor)
    errors = torch.mean((reconstructed - X_new_tensor)**2, dim=1).numpy()

# ========== Classify ==========
# 0 -> Transient (normal), 1 -> No-Transient (anomaly)
predictions = (errors > threshold).astype(int)

# Map to labels
labels = ["Transient" if p==0 else "No-Transient" for p in predictions]

# ========== Summary ==========
for i, label in enumerate(labels):
    print(f"Sample {i+1}: {label}, Reconstruction Error: {errors[i]:.6f}")

# Optional: statistics
num_transient = sum(predictions==0)
num_no_transient = sum(predictions==1)
print(f"\nTotal Samples: {len(predictions)}")
print(f"Classified as Transient: {num_transient}")
print(f"Classified as No-Transient: {num_no_transient}")
