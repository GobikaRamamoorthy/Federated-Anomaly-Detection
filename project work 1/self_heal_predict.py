import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np

# ========== Step 1: Define Model ==========
class GlobalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GlobalClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ========== Step 2: Load Scaler & Model ==========
scaler = joblib.load("scaler.pkl")

input_features = [
    "RMS_Voltage", "Peak_Voltage", "THD", "Duration_ms",
    "DWT_Energy_Level1", "DWT_Energy_Level2", "DWT_Entropy", "Signal_Noise_Ratio_dB"
]

df_new = pd.read_csv("substation_0_new - Copy.csv")
X_new = df_new[input_features]

X_scaled = scaler.transform(X_new)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

input_dim = X_new.shape[1]
num_classes = 5

model = GlobalClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

# ========== Step 3: Make Predictions ==========
with torch.no_grad():
    outputs = model(X_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confs, preds = torch.max(probabilities, 1)

label_map = {
    0: "Normal",
    1: "SAG",
    2: "Swell",
    3: "Transient",
    4: "Harmonics"
}

predicted_classes = [label_map[p.item()] for p in preds]

# ========== Step 4: Self-Healing Logic ==========
def self_heal(row, anomaly_type):
    healed_row = row.copy()
    
    if anomaly_type == "Swell":
        # Swell = voltage too high → reduce voltage & duration
        healed_row["RMS_Voltage"] *= 0.9
        healed_row["Peak_Voltage"] *= 0.9
        healed_row["Duration_ms"] *= 0.8

    elif anomaly_type == "SAG":
        # SAG = Short circuit / voltage dip → increase voltage
        healed_row["RMS_Voltage"] *= 1.1
        healed_row["Peak_Voltage"] *= 1.1

    elif anomaly_type == "Transient":
        # Transient = sudden spike → reduce THD and DWT energy
        healed_row["THD"] *= 0.8
        healed_row["DWT_Energy_Level1"] *= 0.8
        healed_row["DWT_Energy_Level2"] *= 0.8

    elif anomaly_type == "Harmonics":
        # Harmonics = waveform distortion → reduce THD, increase SNR
        healed_row["THD"] *= 0.7
        healed_row["Signal_Noise_Ratio_dB"] *= 1.2

    return healed_row

# Apply healing for anomalies
healed_data = []
for i, row in df_new.iterrows():
    anomaly_type = predicted_classes[i]
    if anomaly_type == "Normal":
        healed_row = row.copy()
    else:
        healed_row = self_heal(row, anomaly_type)
    healed_data.append(healed_row)

df_healed = pd.DataFrame(healed_data)

# ========== Step 5: Combine Results ==========
df_new["Predicted_Class"] = predicted_classes
df_new["Confidence (%)"] = (confs.numpy() * 100).round(2)

# Add healed columns
for col in input_features:
    df_new[f"Healed_{col}"] = df_healed[col]

# ========== Step 6: Display and Save ==========
print("\n✅ Classification + Self-Healing Results:\n")
print(df_new[["Predicted_Class", "Confidence (%)"] + [f"Healed_{c}" for c in input_features]])

df_new.to_csv("substation_self_healed.csv", index=False)
print("\n💾 Results saved as substation_self_healed.csv")
