# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import pickle
# import matplotlib.pyplot as plt

# # ========== Define Autoencoder ==========
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim=16):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, latent_dim),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, input_dim)
#         )

#     def forward(self, x):
#         return self.decoder(self.encoder(x))


# # ========== Load New Test Data ==========
# print("="*70)
# print("TWO-STAGE ANOMALY DETECTION WITH DUAL CLASSIFICATION")
# print("="*70)
# print()

# try:
#     df_new = pd.read_csv("substation_0_new.csv")
#     X_new = df_new.drop(columns=['Label'], errors='ignore').values
#     print(f"✅ Loaded test data: {X_new.shape[0]} samples, {X_new.shape[1]} features")
# except FileNotFoundError:
#     print("❌ Test data file 'substation_0_new.csv' not found!")
#     print("   Please ensure the file exists in the current directory.")
#     exit(1)

# input_dim = X_new.shape[1]

# # ========== STAGE 1: Load Substation 0 Model & Threshold ==========
# print()
# print("="*70)
# print("STAGE 1: LOADING SUBSTATION 0 MODEL (Normal vs Anomaly Detection)")
# print("="*70)
# print()

# try:
#     substation_0_model = Autoencoder(input_dim, latent_dim=16)
#     substation_0_model.load_state_dict(torch.load("substation_0_best_autoencoder.pth"))
#     substation_0_model.eval()
#     print("✅ Substation 0 model loaded: substation_0_best_autoencoder.pth")
# except FileNotFoundError:
#     print("❌ Substation 0 model file 'substation_0_best_autoencoder.pth' not found!")
#     exit(1)

# try:
#     with open('scaler_substation_0.pkl', 'rb') as f:
#         scaler_0 = pickle.load(f)
#     print("✅ Substation 0 scaler loaded: scaler_substation_0.pkl")
# except FileNotFoundError:
#     print("❌ Scaler file 'scaler_substation_0.pkl' not found!")
#     exit(1)

# try:
#     with open('threshold_substation_0.pkl', 'rb') as f:
#         threshold_info_0 = pickle.load(f)
#     threshold_0 = threshold_info_0.get('threshold', 0.01)
#     print(f"✅ Substation 0 threshold loaded: {threshold_0:.6f}")
# except FileNotFoundError:
#     print("❌ Threshold file 'threshold_substation_0.pkl' not found!")
#     exit(1)

# # ========== STAGE 2: Load Global Model & Thresholds ==========
# print()
# print("="*70)
# print("STAGE 2: LOADING GLOBAL(Anomaly Type Classification)")
# print("="*70)
# print()

# try:
#     global_model = Autoencoder(input_dim, latent_dim=16)
#     global_model.load_state_dict(torch.load("global_federated_model.pth"))
#     global_model.eval()
#     print("✅ Global model loaded: global_anomaly_model.pth")
# except FileNotFoundError:
#     print("❌ Global model file 'global_anomaly_model.pth' not found!")
#     exit(1)

# try:
#     with open('global.pkl', 'rb') as f:
#         scaler_global = pickle.load(f)
#     print("✅ Global scaler loaded: scaler_global.pkl")
# except FileNotFoundError:
#     print("❌ Scaler file 'scaler_global.pkl' not found!")
#     exit(1)

# try:
#     with open('threshold_global.pkl', 'rb') as f:
#         thresholds_info = pickle.load(f)
#     local_thresholds = thresholds_info['thresholds']
#     print(f"✅ Global thresholds loaded:")
#     print(f"   Sag: {local_thresholds[0]:.6f}")
#     print(f"   Swell: {local_thresholds[1]:.6f}")
#     print(f"   Transient: {local_thresholds[2]:.6f}")
#     print(f"   Harmonics: {local_thresholds[3]:.6f}")
# except FileNotFoundError:
#     print("❌ Threshold file 'threshold_global.pkl' not found!")
#     exit(1)

# anomaly_types = ["Sag", "Swell", "Transient", "Harmonics"]

# # Load local models for comparison
# print()
# print("Loading local models for comparison...")
# local_model_files = [
#     "substation_1_best_autoencoder.pth",
#     "substation_2_best_autoencoder.pth",
#     "substation_3_best_autoencoder.pth",
#     "substation_4_best_autoencoder.pth"
# ]

# local_models = []
# local_scalers = []
# use_local_models = True

# for i, model_file in enumerate(local_model_files):
#     try:
#         local_model = Autoencoder(input_dim, latent_dim=16)
#         local_model.load_state_dict(torch.load(model_file))
#         local_model.eval()
#         local_models.append(local_model)
        
#         with open(f'scaler_substation_{i+1}.pkl', 'rb') as f:
#             scaler = pickle.load(f)
#         local_scalers.append(scaler)
#         print(f"✅ Substation {i+1} ({anomaly_types[i]}) loaded")
#     except FileNotFoundError as e:
#         print(f"⚠️  Local model {i+1} not found - will use global model only")
#         use_local_models = False
#         break

# # ========== Preprocess Data ==========
# print()
# print("⚙️  Preprocessing test data...")

# X_new_scaled_0 = scaler_0.transform(X_new)
# X_new_tensor_0 = torch.tensor(X_new_scaled_0, dtype=torch.float32)

# X_new_scaled_global = scaler_global.transform(X_new)
# X_new_tensor_global = torch.tensor(X_new_scaled_global, dtype=torch.float32)

# print(f"✅ Data normalized for both Stage 1 and Stage 2")

# # ========== STAGE 1: Classify Normal vs Anomaly ==========
# print()
# print("="*70)
# print("STAGE 1: BINARY CLASSIFICATION (Normal vs Anomaly)")
# print("="*70)
# print()

# with torch.no_grad():
#     recon_0 = substation_0_model(X_new_tensor_0)
#     recon_errors_0 = torch.mean((recon_0 - X_new_tensor_0)**2, dim=1).numpy()

# print(f"✅ Reconstruction Errors (Substation 0 Model):")
# print(f"   Mean: {recon_errors_0.mean():.6f}")
# print(f"   Std:  {recon_errors_0.std():.6f}")
# print(f"   Min:  {recon_errors_0.min():.6f}")
# print(f"   Max:  {recon_errors_0.max():.6f}")
# print(f"   Threshold: {threshold_0:.6f}")

# is_anomaly = recon_errors_0 > threshold_0
# normal_count = (is_anomaly == False).sum()
# anomaly_count = is_anomaly.sum()

# print(f"\n✅ Stage 1 Classification Results:")
# print(f"   Normal samples:  {normal_count} ({100*normal_count/len(is_anomaly):.2f}%)")
# print(f"   Anomaly samples: {anomaly_count} ({100*anomaly_count/len(is_anomaly):.2f}%)")

# # ========== STAGE 2A: Using Global Model Only ==========
# # print()
# # print("="*70)
# # print("STAGE 2A: GLOBAL MODEL CLASSIFICATION")
# # print("="*70)
# # print()

# with torch.no_grad():
#     recon_global = global_model(X_new_tensor_global)
#     recon_errors_global = torch.mean((recon_global - X_new_tensor_global)**2, dim=1).numpy()

# # print(f"✅ Reconstruction Errors (Global Model):")
# # print(f"   Mean: {recon_errors_global.mean():.6f}")
# # print(f"   Std:  {recon_errors_global.std():.6f}")

# predictions_global = []
# confidence_scores_global = []

# # print(f"\n🔍 Classifying using global model only...")

# for idx, is_anom in enumerate(is_anomaly):
#     if is_anom:
#         error = recon_errors_global[idx]
#         threshold_exceedance = np.array([
#             (error - local_thresholds[i]) / local_thresholds[i] if local_thresholds[i] > 0 else error
#             for i in range(4)
#         ])
#         anomaly_idx = np.argmax(threshold_exceedance)
#         predicted_type = anomaly_types[anomaly_idx]
#         confidence = threshold_exceedance[anomaly_idx]
        
#         predictions_global.append(predicted_type)
#         confidence_scores_global.append(max(0, confidence))
#     else:
#         predictions_global.append("Normal")
#         confidence_scores_global.append(1.0)

# # print(f"✅ Global model classification completed")

# # ========== STAGE 2B: Using Local Models ==========
# print()
# print("="*70)
# print("STAGE 2: MODELS CLASSIFICATION")
# print("="*70)
# print()

# predictions_local = []
# confidence_scores_local = []
# local_model_errors = []

# if use_local_models and len(local_models) == 4:
#     print(f"🔍 Computing errors from models...")
    
#     for i, (local_model, local_scaler) in enumerate(zip(local_models, local_scalers)):
#         X_local_scaled = local_scaler.transform(X_new)
#         X_local_tensor = torch.tensor(X_local_scaled, dtype=torch.float32)
        
#         with torch.no_grad():
#             recon_local = local_model(X_local_tensor)
#             errors_local = torch.mean((recon_local - X_local_tensor)**2, dim=1).numpy()
        
#         local_model_errors.append(errors_local)
    
#     local_model_errors = np.array(local_model_errors).T  # Shape: (num_samples, 4)
    
#     # print(f"✅ Local model errors computed\n")
#     # print(f"   Error Statistics per Anomaly Type:")
#     for i, anom_type in enumerate(anomaly_types):
#         print(f"   {anom_type}: mean={local_model_errors[:, i].mean():.6f}, std={local_model_errors[:, i].std():.6f}")
    
#     print(f"\n🔍 Classifying using model...")
    
#     for idx, is_anom in enumerate(is_anomaly):
#         if is_anom:
#             # Get error from each local model for this sample
#             errors_for_sample = local_model_errors[idx]
            
#             # Anomaly type = model with highest error
#             anomaly_idx = np.argmax(errors_for_sample)
#             predicted_type = anomaly_types[anomaly_idx]
#             confidence = errors_for_sample[anomaly_idx]
            
#             predictions_local.append(predicted_type)
#             confidence_scores_local.append(confidence)
#         else:
#             predictions_local.append("Normal")
#             confidence_scores_local.append(1.0)
    
#     print(f"✅  models classification completed")
# else:
#     print(f"⚠️  Local models unavailable - using global model only")
#     predictions_local = predictions_global
#     confidence_scores_local = confidence_scores_global

# # ========== Output Results ==========
# print()
# print("="*70)
# print("FINAL RESULTS")
# print("="*70)
# print()

# results_df = pd.DataFrame(df_new.copy())
# results_df['Reconstruction_Error_Stage1'] = recon_errors_0
# results_df['Reconstruction_Error_Global'] = recon_errors_global
# results_df['Is_Anomaly'] = is_anomaly
# results_df['Predicted_Class_Global'] = predictions_global
# results_df['Confidence_Global'] = confidence_scores_global
# results_df['Predicted_Class'] = predictions_local
# results_df['Confidence_Local'] = confidence_scores_local

# # print("✅ Prediction Summary (Global Model):")
# # print(results_df['Predicted_Class_Global'].value_counts())

# print(f"\n✅ Prediction Summary :")
# print(results_df['Predicted_Class'].value_counts())

# # print(f"\n✅ Agreement between Global and Local predictions:")
# # agreement = (results_df['Predicted_Class_Global'] == results_df['Predicted_Class_Local']).sum()
# # print(f"   {agreement} out of {len(results_df)} samples ({100*agreement/len(results_df):.2f}%) agree")

# # Save results to CSV
# results_df.to_csv("substation_0_new_predictions.csv", index=False)
# print(f"\n✅ Full predictions saved to: substation_0_new_predictions.csv")

# # ========== Visualization ==========
# print("\n📊 Generating visualizations...")

# fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# # Plot 1: Stage 1 Classification
# ax1 = axes[0, 0]
# ax1.hist(recon_errors_0[~is_anomaly], bins=30, alpha=0.7, label='Normal', color='green')
# ax1.hist(recon_errors_0[is_anomaly], bins=30, alpha=0.7, label='Anomaly', color='red')
# ax1.axvline(threshold_0, color='black', linestyle='--', linewidth=2, label='Threshold')
# ax1.set_xlabel('Reconstruction Error')
# ax1.set_ylabel('Count')
# ax1.set_title('Stage 1: Substation 0 Error Distribution')
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# # Plot 2: Global Model Classification
# ax2 = axes[0, 1]
# global_type_counts = {}
# for pred in predictions_global:
#     if pred != "Normal":
#         global_type_counts[pred] = global_type_counts.get(pred, 0) + 1

# if global_type_counts:
#     colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
#     ax2.bar(global_type_counts.keys(), global_type_counts.values(), color=colors_list[:len(global_type_counts)])
#     ax2.set_ylabel('Count')
#     ax2.set_title('Stage 2A: Global Model Anomaly Types')
#     ax2.grid(True, alpha=0.3, axis='y')
# else:
#     ax2.text(0.5, 0.5, 'No Anomalies', ha='center', va='center', fontsize=12)
#     ax2.set_title('Stage 2A: Global Model Anomaly Types')

# # Plot 3: Local Models Classification
# ax3 = axes[0, 2]
# local_type_counts = {}
# for pred in predictions_local:
#     if pred != "Normal":
#         local_type_counts[pred] = local_type_counts.get(pred, 0) + 1

# if local_type_counts:
#     colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
#     ax3.bar(local_type_counts.keys(), local_type_counts.values(), color=colors_list[:len(local_type_counts)])
#     ax3.set_ylabel('Count')
#     ax3.set_title('Stage 2B: Local Models Anomaly Types')
#     ax3.grid(True, alpha=0.3, axis='y')
# else:
#     ax3.text(0.5, 0.5, 'No Anomalies', ha='center', va='center', fontsize=12)
#     ax3.set_title('Stage 2B: Local Models Anomaly Types')

# # Plot 4: Confidence Scores Global vs Local
# ax4 = axes[1, 0]
# anomaly_indices = np.where(is_anomaly)[0]
# if len(anomaly_indices) > 0:
#     ax4.scatter(np.array(confidence_scores_global)[anomaly_indices], 
#                np.array(confidence_scores_local)[anomaly_indices], alpha=0.6, s=50)
#     ax4.plot([0, max(np.array(confidence_scores_global)[anomaly_indices].max(), 
#                      np.array(confidence_scores_local)[anomaly_indices].max())],
#             [0, max(np.array(confidence_scores_global)[anomaly_indices].max(), 
#                     np.array(confidence_scores_local)[anomaly_indices].max())],
#             'r--', alpha=0.5, label='Perfect agreement')
#     ax4.set_xlabel('Global Model Confidence')
#     ax4.set_ylabel('Local Models Confidence')
#     ax4.set_title('Confidence Scores Comparison')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
# else:
#     ax4.text(0.5, 0.5, 'No Anomalies', ha='center', va='center', fontsize=12)
#     ax4.set_title('Confidence Scores Comparison')

# # Plot 6: Overall Summary
# ax6 = axes[1, 2]
# summary_data = {
#     'Normal': normal_count,
#     'Anomalies': anomaly_count
# }
# colors = ['green', 'red']
# ax6.bar(summary_data.keys(), summary_data.values(), color=colors)
# ax6.set_ylabel('Count')
# ax6.set_title('Overall Classification Summary (Stage 1)')
# ax6.grid(True, alpha=0.3, axis='y')

# plt.tight_layout()
# plt.savefig('federated_model_test_results.png', dpi=300, bbox_inches='tight')
# print("✅ Visualization saved to: federated_model_test_results.png")

# plt.show()

# print()
# print("="*70)
# print("✅ TWO-STAGE TEST WITH DUAL CLASSIFICATION COMPLETED")
# print("="*70)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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


# ========== Load New Test Data ==========
print("="*70)
print("TWO-STAGE ANOMALY DETECTION WITH FEDERATED GLOBAL MODEL")
print("="*70)
print()

try:
    df_new = pd.read_csv("substation_0_new.csv")
    X_new = df_new.drop(columns=['Label'], errors='ignore').values
    print(f"✅ Loaded test data: {X_new.shape[0]} samples, {X_new.shape[1]} features")
except FileNotFoundError:
    print("❌ Test data file 'substation_0_new.csv' not found!")
    print("   Please ensure the file exists in the current directory.")
    exit(1)

input_dim = X_new.shape[1]

# ========== STAGE 1: Load Substation 0 Model & Threshold ==========
print()
print("="*70)
print("STAGE 1: LOADING SUBSTATION 0 MODEL (Normal vs Anomaly Detection)")
print("="*70)
print()

try:
    substation_0_model = Autoencoder(input_dim, latent_dim=16)
    substation_0_model.load_state_dict(torch.load("substation_0_best_autoencoder.pth"))
    substation_0_model.eval()
    print("✅ Substation 0 model loaded: substation_0_best_autoencoder.pth")
except FileNotFoundError:
    print("❌ Substation 0 model file 'substation_0_best_autoencoder.pth' not found!")
    exit(1)

try:
    with open('scaler_substation_0.pkl', 'rb') as f:
        scaler_0 = pickle.load(f)
    print("✅ Substation 0 scaler loaded: scaler_substation_0.pkl")
except FileNotFoundError:
    print("❌ Scaler file 'scaler_substation_0.pkl' not found!")
    exit(1)

try:
    with open('threshold_substation_0.pkl', 'rb') as f:
        threshold_info_0 = pickle.load(f)
    threshold_0 = threshold_info_0.get('threshold', 0.01)
    print(f"✅ Substation 0 threshold loaded: {threshold_0:.6f}")
except FileNotFoundError:
    print("❌ Threshold file 'threshold_substation_0.pkl' not found!")
    exit(1)

# ========== STAGE 2: Load Federated Global Model & Thresholds ==========
print()
print("="*70)
print("STAGE 2: LOADING FEDERATED GLOBAL MODEL (Anomaly Type Classification)")
print("="*70)
print()

try:
    global_model = Autoencoder(input_dim, latent_dim=16)
    global_model.load_state_dict(torch.load("global_federated_model.pth"))
    global_model.eval()
    print("✅ Federated global model loaded: global_federated_model.pth")
except FileNotFoundError:
    print("❌ Global model file 'global_federated_model.pth' not found!")
    print("   Run the federated averaging script first to create this model.")
    exit(1)

try:
    with open('global_scaler.pkl', 'rb') as f:
        scaler_global = pickle.load(f)
    print("✅ Global scaler loaded: global_scaler.pkl")
except FileNotFoundError:
    print("❌ Scaler file 'global_scaler.pkl' not found!")
    exit(1)

try:
    with open('global_thresholds.pkl', 'rb') as f:
        global_thresholds = pickle.load(f)
    print(f"✅ Global thresholds loaded:")
    print(f"   Sag (Label 1):       {global_thresholds[1]:.6f}")
    print(f"   Swell (Label 2):     {global_thresholds[2]:.6f}")
    print(f"   Transient (Label 3): {global_thresholds[3]:.6f}")
    print(f"   Harmonic (Label 4):  {global_thresholds[4]:.6f}")
except FileNotFoundError:
    print("❌ Threshold file 'global_thresholds.pkl' not found!")
    exit(1)

anomaly_types = ["Sag", "Swell", "Transient", "Harmonic"]
label_to_name = {1: "Sag", 2: "Swell", 3: "Transient", 4: "Harmonic"}

# ========== Preprocess Data ==========
print()
print("⚙️  Preprocessing test data...")

X_new_scaled_0 = scaler_0.transform(X_new)
X_new_tensor_0 = torch.tensor(X_new_scaled_0, dtype=torch.float32)

X_new_scaled_global = scaler_global.transform(X_new)
X_new_tensor_global = torch.tensor(X_new_scaled_global, dtype=torch.float32)

print(f"✅ Data normalized for both Stage 1 and Stage 2")

# ========== STAGE 1: Classify Normal vs Anomaly ==========
print()
print("="*70)
print("STAGE 1: BINARY CLASSIFICATION (Normal vs Anomaly)")
print("="*70)
print()

with torch.no_grad():
    recon_0 = substation_0_model(X_new_tensor_0)
    recon_errors_0 = torch.mean((recon_0 - X_new_tensor_0)**2, dim=1).numpy()

print(f"✅ Reconstruction Errors (Substation 0 Model):")
print(f"   Mean: {recon_errors_0.mean():.6f}")
print(f"   Std:  {recon_errors_0.std():.6f}")
print(f"   Min:  {recon_errors_0.min():.6f}")
print(f"   Max:  {recon_errors_0.max():.6f}")
print(f"   Threshold: {threshold_0:.6f}")

is_anomaly = recon_errors_0 > threshold_0
normal_count = (is_anomaly == False).sum()
anomaly_count = is_anomaly.sum()

print(f"\n✅ Stage 1 Classification Results:")
print(f"   Normal samples:  {normal_count} ({100*normal_count/len(is_anomaly):.2f}%)")
print(f"   Anomaly samples: {anomaly_count} ({100*anomaly_count/len(is_anomaly):.2f}%)")

# ========== STAGE 2: Classify Anomaly Types using Federated Global Model ==========
print()
print("="*70)
print("STAGE 2: FEDERATED GLOBAL MODEL CLASSIFICATION")
print("="*70)
print()

with torch.no_grad():
    recon_global = global_model(X_new_tensor_global)
    recon_errors_global = torch.mean((recon_global - X_new_tensor_global)**2, dim=1).numpy()

print(f"✅ Reconstruction Errors (Federated Global Model):")
print(f"   Mean: {recon_errors_global.mean():.6f}")
print(f"   Std:  {recon_errors_global.std():.6f}")
print(f"   Min:  {recon_errors_global.min():.6f}")
print(f"   Max:  {recon_errors_global.max():.6f}")

predictions = []
predicted_labels = []
confidence_scores = []

print(f"\n🔍 Classifying anomalies using federated global model...")

for idx, is_anom in enumerate(is_anomaly):
    if is_anom:
        error = recon_errors_global[idx]
        
        # Find closest threshold (minimum distance)
        min_distance = float('inf')
        predicted_label = 1
        
        for label, threshold in global_thresholds.items():
            distance = abs(error - threshold)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label
        
        predicted_type = label_to_name[predicted_label]
        confidence = error / global_thresholds[predicted_label]
        
        predictions.append(predicted_type)
        predicted_labels.append(predicted_label)
        confidence_scores.append(confidence)
    else:
        predictions.append("Normal")
        predicted_labels.append(0)
        confidence_scores.append(1.0)

print(f"✅ Federated global model classification completed")

# ========== Output Results ==========
print()
print("="*70)
print("FINAL RESULTS")
print("="*70)
print()

results_df = pd.DataFrame(df_new.copy())
results_df['Reconstruction_Error_Stage1'] = recon_errors_0
results_df['Reconstruction_Error_Stage2'] = recon_errors_global
results_df['Is_Anomaly'] = is_anomaly
results_df['Predicted_Class'] = predictions
results_df['Predicted_Label'] = predicted_labels
results_df['Confidence_Score'] = confidence_scores

print(f"✅ Prediction Summary:")
print(results_df['Predicted_Class'].value_counts())

# Calculate per-class statistics for anomalies
print(f"\n✅ Anomaly Type Statistics:")
for anom_type in anomaly_types:
    count = (results_df['Predicted_Class'] == anom_type).sum()
    if count > 0:
        avg_confidence = results_df[results_df['Predicted_Class'] == anom_type]['Confidence_Score'].mean()
        print(f"   {anom_type:12s}: {count:4d} samples, Avg Confidence: {avg_confidence:.4f}")

# Save results to CSV
results_df.to_csv("federated_model_predictions.csv", index=False)
print(f"\n✅ Full predictions saved to: federated_model_predictions.csv")

# ========== Visualization ==========
print("\n📊 Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Stage 1 Classification
ax1 = axes[0, 0]
ax1.hist(recon_errors_0[~is_anomaly], bins=30, alpha=0.7, label='Normal', color='green')
ax1.hist(recon_errors_0[is_anomaly], bins=30, alpha=0.7, label='Anomaly', color='red')
ax1.axvline(threshold_0, color='black', linestyle='--', linewidth=2, label='Threshold')
ax1.set_xlabel('Reconstruction Error')
ax1.set_ylabel('Count')
ax1.set_title('Stage 1: Normal vs Anomaly Detection')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Anomaly Type Distribution
ax2 = axes[0, 1]
type_counts = {}
for pred in predictions:
    if pred != "Normal":
        type_counts[pred] = type_counts.get(pred, 0) + 1

if type_counts:
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax2.bar(type_counts.keys(), type_counts.values(), color=colors_list[:len(type_counts)])
    ax2.set_ylabel('Count')
    ax2.set_title('Stage 2: Anomaly Type Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (k, v) in enumerate(type_counts.items()):
        ax2.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'No Anomalies Detected', ha='center', va='center', fontsize=12)
    ax2.set_title('Stage 2: Anomaly Type Distribution')

# Plot 3: Reconstruction Error Distribution per Anomaly Type
ax3 = axes[0, 2]
colors_dict = {'Sag': '#FF6B6B', 'Swell': '#4ECDC4', 'Transient': '#45B7D1', 'Harmonic': '#96CEB4'}
for anom_type in anomaly_types:
    mask = (results_df['Predicted_Class'] == anom_type) & (results_df['Is_Anomaly'] == True)
    if mask.sum() > 0:
        errors = results_df[mask]['Reconstruction_Error_Stage2']
        ax3.hist(errors, bins=20, alpha=0.6, label=anom_type, color=colors_dict[anom_type])

# Add threshold lines
for label, threshold in global_thresholds.items():
    ax3.axvline(threshold, linestyle='--', linewidth=1.5, label=f'{label_to_name[label]} Threshold')

ax3.set_xlabel('Reconstruction Error')
ax3.set_ylabel('Count')
ax3.set_title('Stage 2: Error Distribution by Type')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Confidence Scores Distribution
ax4 = axes[1, 0]
anomaly_mask = results_df['Is_Anomaly'] == True
if anomaly_mask.sum() > 0:
    for anom_type in anomaly_types:
        mask = (results_df['Predicted_Class'] == anom_type) & anomaly_mask
        if mask.sum() > 0:
            confidences = results_df[mask]['Confidence_Score']
            ax4.hist(confidences, bins=15, alpha=0.6, label=anom_type, color=colors_dict[anom_type])
    
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('Count')
    ax4.set_title('Confidence Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No Anomalies', ha='center', va='center', fontsize=12)
    ax4.set_title('Confidence Score Distribution')

# Plot 5: Error Scatter Plot (Stage 1 vs Stage 2)
ax5 = axes[1, 1]
if anomaly_mask.sum() > 0:
    for anom_type in anomaly_types:
        mask = (results_df['Predicted_Class'] == anom_type) & anomaly_mask
        if mask.sum() > 0:
            ax5.scatter(
                results_df[mask]['Reconstruction_Error_Stage1'],
                results_df[mask]['Reconstruction_Error_Stage2'],
                alpha=0.6, s=50, label=anom_type, color=colors_dict[anom_type]
            )
    
    ax5.set_xlabel('Stage 1 Error (Substation 0)')
    ax5.set_ylabel('Stage 2 Error (Global Model)')
    ax5.set_title('Error Comparison: Stage 1 vs Stage 2')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'No Anomalies', ha='center', va='center', fontsize=12)
    ax5.set_title('Error Comparison')

# Plot 6: Overall Summary Pie Chart
ax6 = axes[1, 2]
summary_counts = results_df['Predicted_Class'].value_counts()
colors = ['green'] if 'Normal' in summary_counts.index else []
colors += [colors_dict.get(k, '#CCCCCC') for k in summary_counts.index if k != 'Normal']

ax6.pie(summary_counts.values, labels=summary_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors[:len(summary_counts)])
ax6.set_title('Overall Classification Summary')

plt.tight_layout()
plt.savefig('federated_model_test_results.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to: federated_model_test_results.png")

plt.show()

print()
print("="*70)
print("✅ FEDERATED MODEL TESTING COMPLETED")
print("="*70)
print()
print("Summary:")
print(f"  Total samples tested: {len(results_df)}")
print(f"  Normal samples:       {normal_count} ({100*normal_count/len(results_df):.2f}%)")
print(f"  Anomaly samples:      {anomaly_count} ({100*anomaly_count/len(results_df):.2f}%)")
print()
print("Files generated:")
print("  - federated_model_predictions.csv")
print("  - federated_model_test_results.png")