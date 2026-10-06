import pandas as pd
import torch
import torch.nn as nn
import joblib

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

# ========== Step 2: Load Scaler & New Data ==========
scaler = joblib.load("scaler.pkl")
df_new = pd.read_csv("substation_0_new.csv")

# Ensure column order matches training
X_new = df_new.copy()
X_scaled = scaler.transform(X_new)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ========== Step 3: Load Trained Model ==========
input_dim = X_new.shape[1]
num_classes = 5  # Normal, SAF, Swell, Transient, Harmonics

model = GlobalClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

# ========== Step 4: Make Predictions ==========
with torch.no_grad():
    outputs = model(X_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confs, preds = torch.max(probabilities, 1)

# ========== Step 5: Map Numeric Labels to Class Names ==========
label_map = {
    0: "Normal",
    1: "SAF",
    2: "Swell",
    3: "Transient",
    4: "Harmonics"
}
predicted_labels = [label_map[p.item()] for p in preds]

# ========== Step 6: Add Results to DataFrame ==========
df_new["Predicted_Label"] = preds.numpy()
df_new["Predicted_Class"] = predicted_labels
df_new["Confidence (%)"] = (confs.numpy() * 100).round(2)

# ========== Step 7: Display & Save ==========
print("\n✅ Predictions for substation_new.csv:\n")
print(df_new[["Predicted_Class", "Confidence (%)"]])

df_new.to_csv("substation_new_predictions.csv", index=False)
print("\n💾 Results saved as substation_new_predictions.csv")
