import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== Define Model Architecture (must match training) ==========
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

# ========== Load Model and Preprocessing Artifacts ==========
print("="*70)
print("Loading Trained Model and Preprocessing Artifacts")
print("="*70)

# Load scaler
try:
    with open('scaler_substation_0.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully")
except FileNotFoundError:
    print("✗ Error: scaler_substation_0.pkl not found. Please train the model first.")
    exit(1)

# Load threshold
try:
    with open('threshold_substation_0.pkl', 'rb') as f:
        threshold_info = pickle.load(f)
        threshold = threshold_info['threshold']
        threshold_method = threshold_info['method']
    print(f"✓ Threshold loaded: {threshold:.6f} (Method: {threshold_method})")
except FileNotFoundError:
    print("✗ Error: threshold_substation_0.pkl not found. Please train the model first.")
    exit(1)

# Load model
try:
    # Determine input dimension from scaler
    input_dim = scaler.n_features_in_
    model = ImprovedAutoencoder(input_dim, latent_dim=16)
    model.load_state_dict(torch.load("substation_0_best_autoencoder.pth"))
    model.eval()
    print(f"✓ Model loaded successfully (Input features: {input_dim})")
except FileNotFoundError:
    print("✗ Error: substation_0_best_autoencoder.pth not found. Please train the model first.")
    exit(1)

print()

# ========== Load New Test Data ==========
print("="*70)
print("Loading New Test Data: substation_0_new.csv")
print("="*70)

try:
    test_df = pd.read_csv("substation_0_new.csv")
    print(f"✓ Dataset loaded successfully")
    print(f"  Total samples: {len(test_df)}")
    print(f"  Columns: {list(test_df.columns)}")
except FileNotFoundError:
    print("✗ Error: substation_0_new.csv not found in current directory")
    exit(1)

# Check if Label column exists
has_labels = 'Label' in test_df.columns

if has_labels:
    print(f"\n✓ Label column found")
    label_counts = test_df['Label'].value_counts().sort_index()
    print(f"  Label distribution:")
    for label, count in label_counts.items():
        label_name = "Normal" if label == 0 else "Abnormal"
        print(f"    {label_name} (Label={label}): {count} samples ({100*count/len(test_df):.2f}%)")
    
    # Extract features and labels
    test_features = test_df.drop(columns=['Label'])
    test_labels = test_df['Label'].values
else:
    print(f"\n⚠ Warning: No Label column found. Will predict without evaluation.")
    test_features = test_df
    test_labels = None

print(f"\n  Feature columns: {list(test_features.columns)}")
print(f"  Number of features: {len(test_features.columns)}")

# Validate feature count
if len(test_features.columns) != input_dim:
    print(f"\n✗ Error: Feature mismatch!")
    print(f"  Expected {input_dim} features, but got {len(test_features.columns)}")
    print(f"  Please ensure the test data has the same features as training data")
    exit(1)

print()

# ========== Predict on New Data ==========
print("="*70)
print("Making Predictions")
print("="*70)

# Scale features
test_features_scaled = scaler.transform(test_features.values)

# Convert to tensor
X_test = torch.tensor(test_features_scaled, dtype=torch.float32)

# Get reconstruction and errors
with torch.no_grad():
    reconstructed = model(X_test)
    reconstruction_errors = torch.mean((reconstructed - X_test) ** 2, dim=1).numpy()

# Make predictions
predictions = (reconstruction_errors > threshold).astype(int)

print(f"✓ Predictions completed")
print(f"  Threshold used: {threshold:.6f}")
print(f"  Samples predicted as Normal: {(predictions == 0).sum()} ({100*(predictions == 0).sum()/len(predictions):.2f}%)")
print(f"  Samples predicted as Abnormal: {(predictions == 1).sum()} ({100*(predictions == 1).sum()/len(predictions):.2f}%)")

print(f"\nReconstruction Error Statistics:")
print(f"  Mean: {reconstruction_errors.mean():.6f}")
print(f"  Std:  {reconstruction_errors.std():.6f}")
print(f"  Min:  {reconstruction_errors.min():.6f}")
print(f"  Max:  {reconstruction_errors.max():.6f}")
print(f"  Median: {np.median(reconstruction_errors):.6f}")

# ========== Evaluate if Labels Available ==========
if has_labels:
    print("\n" + "="*70)
    print("Performance Evaluation")
    print("="*70)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\nAccuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Abnormal")
    print(f"Actual Normal    {cm[0][0]:5d}    {cm[0][1]:5d}")
    if len(cm) > 1:  # If abnormal samples exist
        print(f"     Abnormal    {cm[1][0]:5d}    {cm[1][1]:5d}")
        
        # Detailed metrics
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nDetailed Metrics:")
        print(f"  True Positives (TP):  {tp:5d} - Abnormal correctly detected")
        print(f"  True Negatives (TN):  {tn:5d} - Normal correctly identified")
        print(f"  False Positives (FP): {fp:5d} - Normal wrongly flagged as abnormal")
        print(f"  False Negatives (FN): {fn:5d} - Abnormal missed")
        print(f"\n  Precision:    {precision:.4f} ({100*precision:.2f}%)")
        print(f"  Recall:       {recall:.4f} ({100*recall:.2f}%)")
        print(f"  F1-Score:     {f1:.4f}")
        print(f"  Specificity:  {specificity:.4f} ({100*specificity:.2f}%)")
        print(f"  FPR:          {1-specificity:.4f} ({100*(1-specificity):.2f}%)")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions, 
                                  target_names=['Normal', 'Abnormal'],
                                  digits=4))
    else:
        # Only normal samples in test set
        tn, fp = cm[0][0], cm[0][1]
        print(f"\nNote: Test set contains only normal samples")
        print(f"  True Negatives (TN):  {tn:5d} - Normal correctly identified")
        print(f"  False Positives (FP): {fp:5d} - Normal wrongly flagged as abnormal")
        print(f"  False Positive Rate:  {100*fp/len(test_labels):.2f}%")

# ========== Sample-by-Sample Results ==========
print("\n" + "="*70)
print("Sample-by-Sample Predictions (First 20 samples)")
print("="*70)

if has_labels:
    print(f"{'Index':<8} {'True':<10} {'Predicted':<10} {'Error':<12} {'Status'}")
    print("-"*70)
    for i in range(min(20, len(predictions))):
        true_label = "Normal" if test_labels[i] == 0 else "Abnormal"
        pred_label = "Normal" if predictions[i] == 0 else "Abnormal"
        status = "✓ Correct" if predictions[i] == test_labels[i] else "✗ Wrong"
        print(f"{i:<8} {true_label:<10} {pred_label:<10} {reconstruction_errors[i]:<12.6f} {status}")
else:
    print(f"{'Index':<8} {'Predicted':<10} {'Error':<12} {'Confidence'}")
    print("-"*70)
    for i in range(min(20, len(predictions))):
        pred_label = "Normal" if predictions[i] == 0 else "Abnormal"
        # Confidence: how far from threshold
        confidence = abs(reconstruction_errors[i] - threshold) / threshold
        print(f"{i:<8} {pred_label:<10} {reconstruction_errors[i]:<12.6f} {confidence:.4f}")

# ========== Identify High-Risk Samples ==========
if (predictions == 1).any():
    print("\n" + "="*70)
    print("High-Risk Samples (Predicted as Abnormal)")
    print("="*70)
    
    abnormal_indices = np.where(predictions == 1)[0]
    abnormal_errors = reconstruction_errors[abnormal_indices]
    
    # Sort by error (highest first)
    sorted_indices = abnormal_indices[np.argsort(abnormal_errors)[::-1]]
    
    print(f"\nFound {len(abnormal_indices)} abnormal samples")
    print(f"Top 10 most abnormal samples:\n")
    
    if has_labels:
        print(f"{'Index':<8} {'True Label':<12} {'Error':<12} {'Deviation'}")
        print("-"*70)
        for idx in sorted_indices[:10]:
            true_label = "Normal" if test_labels[idx] == 0 else "Abnormal"
            deviation = (reconstruction_errors[idx] - threshold) / threshold * 100
            print(f"{idx:<8} {true_label:<12} {reconstruction_errors[idx]:<12.6f} +{deviation:.1f}%")
    else:
        print(f"{'Index':<8} {'Error':<12} {'Deviation from Threshold'}")
        print("-"*70)
        for idx in sorted_indices[:10]:
            deviation = (reconstruction_errors[idx] - threshold) / threshold * 100
            print(f"{idx:<8} {reconstruction_errors[idx]:<12.6f} +{deviation:.1f}%")

# ========== Save Results ==========
print("\n" + "="*70)
print("Saving Results")
print("="*70)

# Create results DataFrame
results_df = test_df.copy()
results_df['Predicted_Label'] = predictions
results_df['Reconstruction_Error'] = reconstruction_errors
results_df['Anomaly_Score'] = (reconstruction_errors / threshold)  # Normalized score

if has_labels:
    results_df['Correct'] = (predictions == test_labels).astype(int)

# Save to CSV
output_filename = "substation_0_new_predictions.csv"
results_df.to_csv(output_filename, index=False)
print(f"✓ Predictions saved to: {output_filename}")

# Save summary statistics
summary = {
    'total_samples': len(predictions),
    'predicted_normal': int((predictions == 0).sum()),
    'predicted_abnormal': int((predictions == 1).sum()),
    'threshold_used': float(threshold),
    'threshold_method': threshold_method,
    'mean_reconstruction_error': float(reconstruction_errors.mean()),
    'std_reconstruction_error': float(reconstruction_errors.std()),
    'min_reconstruction_error': float(reconstruction_errors.min()),
    'max_reconstruction_error': float(reconstruction_errors.max()),
}

if has_labels:
    summary['accuracy'] = float(accuracy)
    summary['true_label_counts'] = {
        'normal': int((test_labels == 0).sum()),
        'abnormal': int((test_labels == 1).sum())
    }

import json
summary_filename = "substation_0_new_summary.json"
with open(summary_filename, 'w') as f:
    json.dump(summary, f, indent=4)
print(f"✓ Summary saved to: {summary_filename}")

# ========== Visualization ==========
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Reconstruction Error Distribution
    plt.subplot(1, 3, 1)
    if has_labels:
        normal_errors = reconstruction_errors[test_labels == 0]
        if (test_labels == 1).any():
            abnormal_errors = reconstruction_errors[test_labels == 1]
            plt.hist(abnormal_errors, bins=30, alpha=0.7, label='Abnormal', color='red', density=True)
        plt.hist(normal_errors, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    else:
        plt.hist(reconstruction_errors, bins=30, alpha=0.7, label='All Samples', color='gray', density=True)
    
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: Error by Sample
    plt.subplot(1, 3, 2)
    if has_labels:
        normal_mask = test_labels == 0
        abnormal_mask = test_labels == 1
        plt.scatter(np.where(normal_mask)[0], reconstruction_errors[normal_mask], 
                   alpha=0.5, s=20, label='Normal', color='blue')
        if abnormal_mask.any():
            plt.scatter(np.where(abnormal_mask)[0], reconstruction_errors[abnormal_mask], 
                       alpha=0.5, s=20, label='Abnormal', color='red')
    else:
        plt.scatter(range(len(reconstruction_errors)), reconstruction_errors, 
                   alpha=0.5, s=20, label='Samples', color='gray')
    
    plt.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error per Sample')
    plt.legend()
    plt.yscale('log')
    
    # Plot 3: Predictions Distribution
    plt.subplot(1, 3, 3)
    pred_counts = [np.sum(predictions == 0), np.sum(predictions == 1)]
    colors = ['green', 'red']
    plt.bar(['Normal', 'Abnormal'], pred_counts, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    for i, count in enumerate(pred_counts):
        plt.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    viz_filename = 'substation_0_new_predictions.png'
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {viz_filename}")
    
except ImportError:
    print("⚠ matplotlib not available, skipping visualization")

# ========== Final Summary ==========
print("\n" + "="*70)
print("Prediction Complete!")
print("="*70)

print("\n📊 Summary:")
print(f"  Total samples processed: {len(predictions)}")
print(f"  Predicted as Normal: {(predictions == 0).sum()} ({100*(predictions == 0).sum()/len(predictions):.2f}%)")
print(f"  Predicted as Abnormal: {(predictions == 1).sum()} ({100*(predictions == 1).sum()/len(predictions):.2f}%)")

if has_labels:
    print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")

print("\n📁 Output Files:")
print(f"  - {output_filename} (detailed predictions)")
print(f"  - {summary_filename} (summary statistics)")
print(f"  - {viz_filename} (visualization)")

print("\n✅ All done!")