import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ========== Step 1: Load Dataset ==========
df = pd.read_csv("dataset_augmented (1).csv")

# Drop non-numeric / non-feature columns
X = df.drop(columns=["ID", "Fault_Type", "Phase", "Label"])
y = df["Label"]

# ========== Step 2: Normalize features ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for inference
joblib.dump(scaler, "scaler.pkl")

# ========== Step 3: Convert to Tensors ==========
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# Split into train and test sets (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ========== Step 4: Define Model ==========
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

# Instantiate model
input_dim = X.shape[1]
num_classes = len(df["Label"].unique())
model = GlobalClassifier(input_dim, num_classes)

# ========== Step 5: Define Loss & Optimizer ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== Step 6: Train ==========
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# ========== Step 7: Evaluate ==========
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

acc = accuracy_score(y_true, y_pred)
print("\n✅ Test Accuracy:", acc)
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# ========== Step 8: Save the model ==========
torch.save(model.state_dict(), "global_model.pth")
print("\n💾 Model saved as global_model.pth")


import numpy as np
from sklearn.metrics import precision_recall_curve

# ========== Step 9: Compute Optimal Thresholds per Class ==========
print("\n🎯 Computing optimal probability threshold for each class...")

all_probs = []
all_labels = []

# Collect softmax probabilities on test set
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.numpy())
        all_labels.append(labels.numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

thresholds_per_class = {}
num_classes = len(np.unique(all_labels))

for c in range(num_classes):
    y_true_class = (all_labels == c).astype(int)
    y_score_class = all_probs[:, c]

    precision, recall, thr = precision_recall_curve(y_true_class, y_score_class)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    if len(thr) > 0:
        best_thr = thr[np.argmax(f1)]
    else:
        best_thr = 0.5  # fallback if class is missing
    thresholds_per_class[c] = round(float(best_thr), 3)

# ========== Step 10: Print Threshold Summary ==========
label_map = {
    0: "Normal",
    1: "sag",
    2: "Swell",
    3: "Transient",
    4: "Harmonics"
}

print("\n🔹 Optimal Probability Thresholds per Class:")
for c, t in thresholds_per_class.items():
    print(f"Class {c} ({label_map.get(c, 'Unknown')}): {t}")

# Optionally, save them for inference use
joblib.dump(thresholds_per_class, "class_thresholds.pkl")
print("\n💾 Thresholds saved as class_thresholds.pkl")
