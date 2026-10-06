import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler

# ========== Load New Data ==========
df_new = pd.read_csv("substation_0_new - Copy.csv")
X_new = df_new.values  # assuming all columns are features

# ========== Load Scaler ==========
with open("scaler_substation_2.pkl", "rb") as f:
    scaler = pickle.load(f)

X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# ========== Define Autoencoder (must match training) ==========
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
input_dim = X_new_tensor.shape[1]
model = Autoencoder(input_dim, latent_dim=16)
model.load_state_dict(torch.load("substation_2_best_autoencoder.pth"))
model.eval()

# ========== Load Threshold ==========
with open("threshold_substation_2.pkl", "rb") as f:
    threshold_info = pickle.load(f)
threshold = threshold_info['threshold']

# ========== Compute Reconstruction Errors ==========
with torch.no_grad():
    reconstructed = model(X_new_tensor)
    errors = torch.mean((reconstructed - X_new_tensor)**2, dim=1).numpy()

# ========== Classify as Swell / No-Swell ==========
# If reconstruction error <= threshold => Swell (normal)
# If reconstruction error > threshold => No-Swell (anomaly)
predictions = np.where(errors <= threshold, "Swell", "No-Swell")

# ========== Output Results ==========
df_results = df_new.copy()
df_results['Reconstruction_Error'] = errors
df_results['Prediction'] = predictions

print(df_results)

# Optionally save results
df_results.to_csv("substation_2_new_predictions.csv", index=False)
print("\nPredictions saved to 'substation_2_new_predictions.csv'")
