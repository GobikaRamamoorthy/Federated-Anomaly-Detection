# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np

# # ------------------ Autoencoder class (must match training) ------------------
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(16, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, input_dim),
#         )

#     def forward(self, x):
#         return self.decoder(self.encoder(x))

# # ------------------ Anomaly detection function ------------------
# def detect_anomalies(model, data, threshold=0.01):
#     model.eval()
#     with torch.no_grad():
#         recon = model(data)
#         loss = ((recon - data) ** 2).mean(dim=1)
#         anomalies = loss > threshold
#     return anomalies.numpy(), loss.numpy()

# # ------------------ Data loader ------------------
# def load_substation_data(path):
#     df = pd.read_csv(path)
#     return torch.tensor(df.values, dtype=torch.float32)

# # ------------------ Main evaluation loop ------------------
# for i in range(5):
#     data_path = f"substation_{i}_aug.csv"
#     data = load_substation_data(data_path)

#     input_dim = data.shape[1]

#     # Create model instance and load trained weights
#     model = Autoencoder(input_dim)
#     model.load_state_dict(torch.load(f"substation_{i}_global.pth", map_location="cpu"))

#     print(f"✅ Substation {i}: Loaded global model from substation_{i}_global.pth")

#     # Detect anomalies
#     anomalies, losses = detect_anomalies(model, data)

#     # Save results
#     results_df = pd.DataFrame({
#         'Anomaly': anomalies,
#         'Reconstruction Loss': losses
#     })
#     results_df.to_csv(f"substation_{i}_anomaly_results.csv", index=False)

#     print(f"Substation {i} results saved to substation_{i}_anomaly_results.csv\n")




import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# ================== Autoencoder (must match training architecture) ==================
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ================== Compute reconstruction errors ==================
def compute_errors(model, X_array, batch_size=256, device="cpu"):
    model.to(device)
    model.eval()
    X_tensor = torch.tensor(X_array, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    errors = []
    loss_fn = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        for (xb,) in loader:
            recon = model(xb)
            per_sample = loss_fn(recon, xb).mean(dim=1)
            errors.extend(per_sample.cpu().numpy().tolist())
    return np.array(errors)

# ================== Detect anomalies using threshold ==================
def detect_anomalies(errors, threshold):
    preds = (errors > threshold).astype(int)
    return preds

# ================== Evaluate one substation ==================
def evaluate_substation(model, csv_file, threshold=0.01):
    df = pd.read_csv(csv_file)
    if "Label" not in df.columns:
        raise ValueError(f"CSV {csv_file} must contain 'Label' column.")
    feature_cols = [c for c in df.columns if c.lower() not in ["label", "id", "phase", "fault_type"]]
    X = df[feature_cols].values
    y_true = (df["Label"].values != 0).astype(int)  # binary: 0 = normal, 1 = fault

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # compute errors
    errors = compute_errors(model, X_scaled)
    preds = detect_anomalies(errors, threshold)

    # metrics
    acc = accuracy_score(y_true, preds) * 100
    bal = balanced_accuracy_score(y_true, preds) * 100
    prec = precision_score(y_true, preds, zero_division=0) * 100
    rec = recall_score(y_true, preds, zero_division=0) * 100
    f1 = f1_score(y_true, preds, zero_division=0) * 100

    summary = {
        "File": os.path.basename(csv_file),
        "Accuracy": acc,
        "BalancedAcc": bal,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Total Samples": len(y_true),
        "Anomalies Detected": int(preds.sum())
    }
    return summary, errors, preds

# ================== Main Evaluation ==================
if __name__ == "__main__":
    device = "cpu"  # change to "cuda" if GPU available

    # folder containing substation CSVs
    substation_files = sorted(glob.glob("substation_*_aug.csv"))
    if len(substation_files) == 0:
        raise SystemExit("No substation CSV files found.")

    # infer input dimension from first file
    sample_df = pd.read_csv(substation_files[0])
    feature_cols = [c for c in sample_df.columns if c.lower() not in ["label", "id", "phase", "fault_type"]]
    input_dim = len(feature_cols)

    # load trained global model
    model_path = "federated_autoencoder_warmstart.pth"
    if not os.path.exists(model_path):
        raise SystemExit(f"Global model not found at {model_path}")

    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Global model loaded successfully!")

    # evaluation loop
    summaries = []
    total_correct = 0
    total_samples = 0
    threshold = 0.01  # can be tuned based on your dataset

    for csv_file in substation_files:
        summary, errors, preds = evaluate_substation(model, csv_file, threshold)
        summaries.append(summary)
        total_correct += int((np.array(preds) == (pd.read_csv(csv_file)["Label"].values != 0)).sum())
        total_samples += len(preds)

        print(f"\n📊 {summary['File']} Summary:")
        for k, v in summary.items():
            print(f"   {k}: {v}")

    overall_acc = 100.0 * total_correct / total_samples
    print(f"\n🌍 Overall accuracy across all substations: {overall_acc:.2f}%")

    # save summary CSV
    pd.DataFrame(summaries).to_csv("evaluation_summary.csv", index=False)
    print("\n📋 Evaluation summary saved to evaluation_summary.csv")
