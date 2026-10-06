import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# ========== Load New Data ==========
new_data_file = "substation_0_new - Copy.csv"
df_new = pd.read_csv(new_data_file)

# Drop Label if present
if 'Label' in df_new.columns:
    df_new = df_new.drop(columns=['Label'])

# ========== Load Scaler ==========
with open('scaler_substation_1.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Normalize new data
X_new_scaled = scaler.transform(df_new.values)
X_new = torch.tensor(X_new_scaled, dtype=torch.float32)

# ========== Define Autoencoder (same architecture as training) ==========
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

# ========== Load Model ==========
input_dim = X_new.shape[1]
model = Autoencoder(input_dim, latent_dim=16)
model.load_state_dict(torch.load("substation_1_best_autoencoder.pth"))
model.eval()

# ========== Load Threshold ==========
with open('threshold_substation_1.pkl', 'rb') as f:
    threshold_info = pickle.load(f)
threshold = threshold_info['threshold']

# ========== Compute Reconstruction Errors ==========
with torch.no_grad():
    reconstructions = model(X_new)
    errors = torch.mean((reconstructions - X_new) ** 2, dim=1).numpy()

# ========== Classify ==========
# 0 = Sag (Normal), 1 = No Sag (Abnormal)
predictions = (errors > threshold).astype(int)

# Add predictions to dataframe
df_new['Prediction'] = ['Sag' if p==0 else 'No Sag' for p in predictions]
df_new['Reconstruction_Error'] = errors

# ========== Summary ==========
total_samples = len(df_new)
num_sag = np.sum(predictions==0)
num_no_sag = np.sum(predictions==1)

print(f"Total samples: {total_samples}")
print(f"Sag (Normal) samples: {num_sag} ({100*num_sag/total_samples:.2f}%)")
print(f"No Sag (Abnormal) samples: {num_no_sag} ({100*num_no_sag/total_samples:.2f}%)")

# ========== Save results ==========
df_new.to_csv("substation_1_new_predictions.csv", index=False)
print("Predictions saved to 'substation_1_new_predictions.csv'")
