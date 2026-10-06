import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# ========== Load New Data ==========
df_new = pd.read_csv("substation_0_new.csv")
print(f"New data shape: {df_new.shape}")

# Assuming new data has same feature columns as training data
X_new = df_new.values.astype(np.float32)

# ========== Load Scaler ==========
with open('scaler_substation_4.pkl', 'rb') as f:
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

# ========== Load Trained Model ==========
input_dim = X_new.shape[1]
model = Autoencoder(input_dim, latent_dim=16)
model.load_state_dict(torch.load("substation_4_best_autoencoder.pth"))
model.eval()

# ========== Load Threshold ==========
with open('threshold_substation_4.pkl', 'rb') as f:
    threshold_info = pickle.load(f)

threshold = threshold_info['threshold']
print(f"Using threshold: {threshold:.6f}")

# ========== Compute Reconstruction Errors ==========
with torch.no_grad():
    reconstructions = model(X_new_tensor)
    errors = torch.mean((reconstructions - X_new_tensor)**2, dim=1).numpy()

# ========== Classify ==========
# 0 = Harmonics (normal), 1 = No-Harmonics (anomaly)
predictions = (errors > threshold).astype(int)

df_new['Reconstruction_Error'] = errors
df_new['Prediction'] = predictions
df_new['Prediction_Label'] = df_new['Prediction'].map({0: 'Harmonics', 1: 'No-Harmonics'})

# ========== Summary ==========
num_harmonics = (predictions == 0).sum()
num_no_harmonics = (predictions == 1).sum()
print(f"\nClassification Summary:")
print(f"  Harmonics (Normal): {num_harmonics}")
print(f"  No-Harmonics (Anomaly): {num_no_harmonics}")

# ========== Save Results ==========
df_new.to_csv("substation_4_new_classified.csv", index=False)
print("\nResults saved to 'substation_4_new_classified.csv'")
