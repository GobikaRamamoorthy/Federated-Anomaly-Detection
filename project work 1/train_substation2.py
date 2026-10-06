import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# ========== Load and Prepare Data ==========
df = pd.read_csv("substation_2_aug.csv")

# All data is Swell (Label=2)
swell_data = df.drop(columns=['Label']).values
print(f"Total Swell samples: {len(swell_data)}")
print("Note: No 'No-Swell' samples found. Model will train on Swell data only.\n")

# ========== Train/Validation/Test Split ==========
train_swell, temp_swell = train_test_split(swell_data, test_size=0.3, random_state=42)
val_swell, test_swell = train_test_split(temp_swell, test_size=0.5, random_state=42)

print(f"Training set: {len(train_swell)} samples")
print(f"Validation set: {len(val_swell)} samples")
print(f"Test set: {len(test_swell)} samples\n")

# ========== Normalize Features ==========
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_swell)
val_scaled = scaler.transform(val_swell)
test_scaled = scaler.transform(test_swell)

# Save scaler
with open('scaler_substation_2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Convert to torch tensors
X_train = torch.tensor(train_scaled, dtype=torch.float32)
X_val = torch.tensor(val_scaled, dtype=torch.float32)
X_test = torch.tensor(test_scaled, dtype=torch.float32)

# DataLoader
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

# ========== Define Autoencoder ==========
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
input_dim = X_train.shape[1]
model = Autoencoder(input_dim, latent_dim=16)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ========== Train Autoencoder ==========
epochs = 100
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 15

train_losses, val_losses = [], []

print("="*60)
print("Training Autoencoder on Swell Data Only")
print("="*60)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        xb = batch[0]
        optimizer.zero_grad()
        reconstructed = model(xb)
        loss = criterion(reconstructed, xb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), X_val).item()
        val_losses.append(val_loss)
    
    scheduler.step(val_loss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "substation_2_best_autoencoder.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("substation_2_best_autoencoder.pth"))
print(f"\nBest Validation Loss: {best_val_loss:.6f}")

# ========== Compute Reconstruction Errors ==========
model.eval()
with torch.no_grad():
    train_errors = torch.mean((model(X_train) - X_train)**2, dim=1).numpy()
    val_errors = torch.mean((model(X_val) - X_val)**2, dim=1).numpy()
    test_errors = torch.mean((model(X_test) - X_test)**2, dim=1).numpy()

# ========== Determine Threshold ==========
threshold = np.percentile(train_errors, 99)  # conservative threshold
print(f"\nSelected Threshold (99th percentile of train errors): {threshold:.6f}")

# Save threshold
threshold_info = {
    'threshold': threshold,
    'method': '99th Percentile',
    'train_error_mean': train_errors.mean(),
    'train_error_std': train_errors.std()
}
with open('threshold_substation_2.pkl', 'wb') as f:
    pickle.dump(threshold_info, f)

# ========== Test Set Evaluation ==========
predictions = (test_errors > threshold).astype(int)
false_positives = predictions.sum()
true_negatives = len(predictions) - false_positives
accuracy = true_negatives / len(predictions)

print(f"\nTest Set (Swell samples only):")
print(f"  Total samples: {len(predictions)}")
print(f"  Classified as Swell (True Negatives): {true_negatives} ({100*true_negatives/len(predictions):.2f}%)")
print(f"  Classified as Abnormal (False Positives): {false_positives} ({100*false_positives/len(predictions):.2f}%)")
print(f"Accuracy: {accuracy:.4f}")
print(f"False Positive Rate: {100*false_positives/len(predictions):.2f}%")

# ========== Visualization ==========
plt.figure(figsize=(12,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Autoencoder Training History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.figure(figsize=(8,5))
plt.hist(train_errors, bins=50, alpha=0.7, label='Train Errors')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.title('Reconstruction Error Distribution (Training Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
