# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import json
# import os

# # ==============================
# # 1. Autoencoder Definition
# # ==============================
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, input_dim)
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # ==============================
# # 2. Load Global Model
# # ==============================
# SUBSTATION_ID = 0
# MODEL_PATH = f"substation_{SUBSTATION_ID}_global.pth"
# DATA_PATH = f"substation_{SUBSTATION_ID}_new.csv"

# print(f"🔍 Loading model for Substation {SUBSTATION_ID}...")

# if not os.path.exists(DATA_PATH):
#     raise FileNotFoundError(f"⚠️ File not found: {DATA_PATH}")

# data = pd.read_csv(DATA_PATH)
# feature_columns = [col for col in data.columns if col != "Label"]
# inputs = torch.tensor(data[feature_columns].values, dtype=torch.float32)
# input_dim = inputs.shape[1]

# # Load model
# model = Autoencoder(input_dim=input_dim)
# state_dict = torch.load(MODEL_PATH, map_location="cpu")
# model.load_state_dict(state_dict)
# model.eval()
# print("✅ Model loaded successfully!")

# # ==============================
# # 3. Compute Reconstruction Error
# # ==============================
# with torch.no_grad():
#     reconstructed = model(inputs)
#     errors = torch.mean((inputs - reconstructed) ** 2, dim=1).numpy()

# # ==============================
# # 4. Threshold
# # ==============================
# threshold = np.percentile(errors, 95)
# anomalies = errors > threshold
# anomaly_indices = np.where(anomalies)[0]

# # ==============================
# # 5. Classify Anomaly Type using deviation from normal
# # ==============================
# ref_mean = data[feature_columns].mean()  # reference normal behavior

# def classify_anomaly(row):
#     diff = row - ref_mean
#     mean_diff = diff.mean()
#     std_diff = diff.std()
    
#     if mean_diff < -0.5:             # Sag: drop in voltage
#         return 1, "Sag"
#     elif mean_diff > 0.5:            # Swell: rise in voltage
#         return 2, "Swell"
#     elif std_diff > 1.0:             # Transient: sudden spike
#         return 3, "Transient"
#     elif (diff.abs().mean() > 0.2) and (std_diff < 0.5):  # Harmonics: waveform distortion
#         return 4, "Harmonics"
#     else:
#         return 0, "Normal"

# row_reports = []
# for i in range(len(inputs)):
#     is_anomaly = errors[i] > threshold
#     label, predicted_type = classify_anomaly(data.iloc[i][feature_columns])
#     row_reports.append({
#         "row": i,
#         "error": float(errors[i]),
#         "anomaly": bool(is_anomaly),
#         "predicted_label": label,
#         "predicted_type": predicted_type
#     })
#     print(f"Row {i}: Error={errors[i]:.3f}, Anomaly={is_anomaly}, Type={predicted_type}")

# # ==============================
# # 6. Substation Status
# # ==============================
# status = "Corrupted" if len(anomaly_indices) > 0 else "Healthy"

# # ==============================
# # 7. Generate Report
# # ==============================
# report = {
#     "substation_id": SUBSTATION_ID,
#     "status": status,
#     "total_samples": len(inputs),
#     "threshold": float(threshold),
#     "anomaly_count": int(len(anomaly_indices)),
#     "anomaly_percentage": round(len(anomaly_indices) / len(inputs) * 100, 3),
#     "rows": row_reports
# }

# REPORT_PATH = f"substation_{SUBSTATION_ID}_detailed_report.json"
# with open(REPORT_PATH, "w") as f:
#     json.dump(report, f, indent=4)

# print(f"\n📡 Substation status: {status}")
# print(f"📋 Report saved as {REPORT_PATH}")





import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# ========== Load mixed dataset ==========
mixed_csv = r"substation_0_new.csv"
df = pd.read_csv(mixed_csv)

# Keep only feature columns (exclude Label if present)
feature_columns = [col for col in df.columns if col != 'Label']
X = torch.tensor(df[feature_columns].values, dtype=torch.float32)

# ========== Define Autoencoder ==========
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model with correct input_dim
input_dim = X.shape[1]
model = Autoencoder(input_dim)

# ========== Load your trained model ==========
# Make sure this model was trained with the same number of features
trained_model_path = r"substation_1_autoencoder.pth"
model.load_state_dict(torch.load(trained_model_path))
model.eval()

# ========== Detect Anomalies ==========
with torch.no_grad():
    reconstructed = model(X)
    mse_per_sample = ((reconstructed - X) ** 2).mean(dim=1).numpy()
    
# Dynamic threshold
threshold = mse_per_sample.mean() + mse_per_sample.std()
print(f"Dynamic threshold: {threshold:.4f}")

# Add results to dataframe
df['Reconstruction_Error'] = mse_per_sample
df['Anomaly'] = np.where(df['Reconstruction_Error'] > threshold, 'Anomaly', 'Normal')

# Simple rule-based mapping (example)
# You can replace this with proper rules based on RMS, THD, etc.
def detect_type(row):
    if row['Anomaly'] == 'Normal':
        return 'Normal'
    # Example: simple heuristic based on RMS voltage
    if row['RMS_Voltage'] < 200:
        return 'Sag'
    elif row['RMS_Voltage'] > 400:
        return 'Swell'
    elif row['THD'] > 10:
        return 'Harmonics'
    else:
        return 'Transient'

df['Anomaly_Type'] = df.apply(detect_type, axis=1)

# ========== Save output ==========
output_csv = r"detection_output.csv"
df.to_csv(output_csv, index=False)
print(f"✅ Detection results saved to {output_csv}")