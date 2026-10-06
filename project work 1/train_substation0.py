
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
df = pd.read_csv("substation_0_aug.csv")

# Check label distribution
label_counts = df['Label'].value_counts()
print("Label Distribution:")
print(label_counts)
print()

# Get normal data (Label=0)
normal_data = df[df['Label'] == 0].drop(columns=['Label']).values
print(f"Total Normal samples: {len(normal_data)}")

# Check if abnormal data exists
has_abnormal = 1 in df['Label'].values
if has_abnormal:
    abnormal_data = df[df['Label'] == 1].drop(columns=['Label']).values
    print(f"Total Abnormal samples: {len(abnormal_data)}")
else:
    print("No abnormal samples found in dataset.")
    print("The model will be trained on normal data only.")
    print("Threshold will be set using statistical methods on reconstruction errors.")

print()

# ========== Train/Validation Split ==========
# Split normal data: 70% train, 15% validation, 15% test
train_normal, temp_normal = train_test_split(normal_data, test_size=0.3, random_state=42)
val_normal, test_normal = train_test_split(temp_normal, test_size=0.5, random_state=42)

print(f"Training set (normal): {len(train_normal)} samples")
print(f"Validation set (normal): {len(val_normal)} samples")
print(f"Test set (normal): {len(test_normal)} samples")

# If abnormal data exists, split it too
if has_abnormal:
    val_abnormal, test_abnormal = train_test_split(abnormal_data, test_size=0.5, random_state=42)
    print(f"Validation set (abnormal): {len(val_abnormal)} samples")
    print(f"Test set (abnormal): {len(test_abnormal)} samples")

print()

# ========== Normalize Features ==========
scaler = StandardScaler()
train_normal_scaled = scaler.fit_transform(train_normal)
val_normal_scaled = scaler.transform(val_normal)
test_normal_scaled = scaler.transform(test_normal)

if has_abnormal:
    val_abnormal_scaled = scaler.transform(val_abnormal)
    test_abnormal_scaled = scaler.transform(test_abnormal)

# Save scaler
with open('scaler_substation_0.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Convert to torch tensors
X_train = torch.tensor(train_normal_scaled, dtype=torch.float32)
X_val_normal = torch.tensor(val_normal_scaled, dtype=torch.float32)
X_test_normal = torch.tensor(test_normal_scaled, dtype=torch.float32)

if has_abnormal:
    X_val_abnormal = torch.tensor(val_abnormal_scaled, dtype=torch.float32)
    X_test_abnormal = torch.tensor(test_abnormal_scaled, dtype=torch.float32)

# Create DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ========== Define Improved Autoencoder ==========
class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(ImprovedAutoencoder, self).__init__()
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model
input_dim = X_train.shape[1]
model = ImprovedAutoencoder(input_dim, latent_dim=16)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ========== Train Autoencoder ==========
epochs = 100
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 15

print("="*60)
print("Training Autoencoder on Normal Data Only")
print("="*60)

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        xb = batch[0]
        reconstructed = model(xb)
        loss = criterion(reconstructed, xb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_recon = model(X_val_normal)
        val_loss = criterion(val_recon, X_val_normal).item()
        val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "substation_0_best_autoencoder.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("substation_0_best_autoencoder.pth"))
print(f"\nBest Validation Loss: {best_val_loss:.6f}")

# ========== Calculate Reconstruction Errors ==========
model.eval()
with torch.no_grad():
    # Training errors
    train_recon = model(X_train)
    train_errors = torch.mean((train_recon - X_train) ** 2, dim=1).numpy()
    
    # Validation errors
    val_normal_recon = model(X_val_normal)
    val_normal_errors = torch.mean((val_normal_recon - X_val_normal) ** 2, dim=1).numpy()
    
    # Test errors (normal)
    test_normal_recon = model(X_test_normal)
    test_normal_errors = torch.mean((test_normal_recon - X_test_normal) ** 2, dim=1).numpy()
    
    if has_abnormal:
        val_abnormal_recon = model(X_val_abnormal)
        val_abnormal_errors = torch.mean((val_abnormal_recon - X_val_abnormal) ** 2, dim=1).numpy()
        
        test_abnormal_recon = model(X_test_abnormal)
        test_abnormal_errors = torch.mean((test_abnormal_recon - X_test_abnormal) ** 2, dim=1).numpy()

# ========== Find Optimal Threshold ==========
print("\n" + "="*60)
print("Determining Optimal Threshold")
print("="*60)

print(f"\nReconstruction Error Statistics (Training Normal Data):")
print(f"  Mean: {train_errors.mean():.6f}")
print(f"  Std:  {train_errors.std():.6f}")
print(f"  Min:  {train_errors.min():.6f}")
print(f"  Max:  {train_errors.max():.6f}")
print(f"  Median: {np.median(train_errors):.6f}")

# Define multiple threshold strategies
thresholds = {
    'Mean + 2*Std': train_errors.mean() + 2 * train_errors.std(),
    'Mean + 3*Std': train_errors.mean() + 3 * train_errors.std(),
    '95th Percentile': np.percentile(train_errors, 95),
    '99th Percentile': np.percentile(train_errors, 99),
    '99.5th Percentile': np.percentile(train_errors, 99.5),
}

print(f"\nThreshold Candidates:")
for name, value in thresholds.items():
    print(f"  {name:20s}: {value:.6f}")

# If abnormal data exists, evaluate thresholds
if has_abnormal:
    print("\n" + "="*60)
    print("Evaluating Thresholds with Validation Data")
    print("="*60)
    
    # Combine validation data
    all_val_errors = np.concatenate([val_normal_errors, val_abnormal_errors])
    all_val_labels = np.concatenate([np.zeros(len(val_normal_errors)), 
                                     np.ones(len(val_abnormal_errors))])
    
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(all_val_labels, all_val_errors)
    optimal_idx = np.argmax(tpr - fpr)
    thresholds['ROC Optimal'] = roc_thresholds[optimal_idx]
    
    best_f1 = 0
    best_threshold_name = None
    best_threshold_value = None
    
    for name, threshold in thresholds.items():
        predictions = (all_val_errors > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_val_labels, predictions).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(all_val_labels)
        
        print(f"\n{name}: {threshold:.6f}")
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_name = name
            best_threshold_value = threshold
    
    print(f"\n{'='*60}")
    print(f"Best Threshold: {best_threshold_name} = {best_threshold_value:.6f}")
    print(f"Best F1-Score: {best_f1:.4f}")
    print(f"{'='*60}")
    
    auc = roc_auc_score(all_val_labels, all_val_errors)
    print(f"Validation AUC-ROC: {auc:.4f}")
else:
    # No abnormal data - use conservative threshold
    best_threshold_name = '99th Percentile'
    best_threshold_value = thresholds[best_threshold_name]
    
    print(f"\n{'='*60}")
    print(f"Selected Threshold: {best_threshold_name} = {best_threshold_value:.6f}")
    print(f"{'='*60}")
    print("\nNote: Without abnormal samples, using conservative statistical threshold.")
    print("Adjust threshold based on your false positive tolerance:")
    print(f"  - More sensitive (95th percentile): {thresholds['95th Percentile']:.6f}")
    print(f"  - Balanced (99th percentile): {thresholds['99th Percentile']:.6f}")
    print(f"  - More specific (99.5th percentile): {thresholds['99.5th Percentile']:.6f}")

# Save threshold
threshold_info = {
    'threshold': best_threshold_value,
    'method': best_threshold_name,
    'train_error_mean': train_errors.mean(),
    'train_error_std': train_errors.std(),
    'all_thresholds': thresholds
}
with open('threshold_substation_0.pkl', 'wb') as f:
    pickle.dump(threshold_info, f)

# ========== Test Set Evaluation ==========
print("\n" + "="*60)
print("Final Test Set Evaluation")
print("="*60)

if has_abnormal:
    # Combine test data
    all_test_errors = np.concatenate([test_normal_errors, test_abnormal_errors])
    all_test_labels = np.concatenate([np.zeros(len(test_normal_errors)), 
                                      np.ones(len(test_abnormal_errors))])
    
    predictions = (all_test_errors > best_threshold_value).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(all_test_labels, predictions, 
                              target_names=['Normal', 'Abnormal'],
                              digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_test_labels, predictions)
    print(f"                Predicted")
    print(f"              Normal  Abnormal")
    print(f"Actual Normal    {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"     Abnormal    {cm[1][0]:5d}    {cm[1][1]:5d}")
    
    test_auc = roc_auc_score(all_test_labels, all_test_errors)
    print(f"\nTest AUC-ROC Score: {test_auc:.4f}")
else:
    # Only normal test data
    predictions = (test_normal_errors > best_threshold_value).astype(int)
    false_positives = predictions.sum()
    true_negatives = len(predictions) - false_positives
    
    # Calculate accuracy (for normal samples, correct classification is 0)
    accuracy = true_negatives / len(predictions)
    
    print(f"\nTest Set (Normal samples only):")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Classified as Normal (True Negatives): {true_negatives} ({100*true_negatives/len(predictions):.2f}%)")
    print(f"  Classified as Abnormal (False Positives): {false_positives} ({100*false_positives/len(predictions):.2f}%)")
    print(f"\nAccuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
    print(f"False Positive Rate: {100*false_positives/len(predictions):.2f}%")
    
    # Create confusion matrix for normal-only data
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Abnormal")
    print(f"Actual Normal    {true_negatives:5d}    {false_positives:5d}")
    print(f"\nNote: Since test set contains only normal samples:")
    print(f"  - True Negatives (TN): {true_negatives} - Correctly identified as normal")
    print(f"  - False Positives (FP): {false_positives} - Incorrectly flagged as abnormal")
    print(f"  - True Positives (TP): N/A - No abnormal samples in test set")
    print(f"  - False Negatives (FN): N/A - No abnormal samples in test set")

# ========== Visualization ==========
plt.figure(figsize=(15, 5))

# Plot 1: Training History
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Reconstruction Error Distribution
plt.subplot(1, 3, 2)
plt.hist(train_errors, bins=50, alpha=0.7, label='Train', color='blue', density=True)
plt.hist(val_normal_errors, bins=50, alpha=0.7, label='Val Normal', color='green', density=True)
if has_abnormal:
    plt.hist(val_abnormal_errors, bins=50, alpha=0.7, label='Val Abnormal', color='red', density=True)
plt.axvline(best_threshold_value, color='black', linestyle='--', linewidth=2, 
           label=f'Threshold ({best_threshold_name})')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Density')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.yscale('log')

# Plot 3: Error Scatter
plt.subplot(1, 3, 3)
plt.scatter(range(len(test_normal_errors)), sorted(test_normal_errors), 
           alpha=0.6, s=10, label='Test Normal', color='blue')
if has_abnormal:
    plt.scatter(range(len(test_normal_errors), len(test_normal_errors) + len(test_abnormal_errors)), 
               sorted(test_abnormal_errors), alpha=0.6, s=10, label='Test Abnormal', color='red')
plt.axhline(best_threshold_value, color='black', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Sample Index (sorted)')
plt.ylabel('Reconstruction Error')
plt.title('Test Set Reconstruction Errors')
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.savefig('autoencoder_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'autoencoder_evaluation.png'")

# ========== Summary ==========
print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print("\n✓ Saved files:")
print("  - substation_0_best_autoencoder.pth (model weights)")
print("  - scaler_substation_0.pkl (feature scaler)")
print("  - threshold_substation_0.pkl (classification threshold)")
print("  - autoencoder_evaluation.png (visualization)")

print("\n📊 Model Summary:")
print(f"  - Input features: {input_dim}")
print(f"  - Latent dimension: 16")
print(f"  - Training samples: {len(train_normal)}")
print(f"  - Best validation loss: {best_val_loss:.6f}")
print(f"  - Classification threshold: {best_threshold_value:.6f}")

print("\n🎯 Next Steps:")
print("  1. Run 'predict_autoencoder.py' to test on new data")
print("  2. Adjust threshold if needed based on your false positive tolerance")
print("  3. Collect abnormal samples to improve threshold selection")