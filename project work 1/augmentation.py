import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ==== Step 1: Load combined dataset ====
df = pd.read_csv("dataset_augmented (1).csv")  # your combined dataset
print("✅ Original dataset shape:", df.shape)

# ==== Step 2: Separate features and labels ====
non_numeric_cols = ['ID', 'Fault_Type', 'Phase']  # adjust if needed
X = df.drop(columns=non_numeric_cols + ['Label'], errors='ignore')
y = df['Label']

# ==== Step 3: Normalize numeric features ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==== Step 4: Apply SMOTE to increase dataset size ====
target_per_class = 3000  # <-- change as needed
sampling_strategy = {cls: target_per_class for cls in np.unique(y)}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("✅ Dataset after SMOTE:", X_resampled.shape, y_resampled.shape)

# ==== Step 5: Convert back to original scale ====
X_resampled = scaler.inverse_transform(X_resampled)
new_df = pd.DataFrame(X_resampled, columns=X.columns)
new_df['Label'] = y_resampled

# ==== Step 6: Add optional noise for diversity ====
noise_factor = 0.05  # adjust as needed
for col in X.columns:
    new_df[col] += np.random.normal(0, noise_factor * new_df[col].std(), len(new_df))

# ==== Step 7: Save augmented dataset ====
new_df.to_csv("all_faults_augmented.csv", index=False)
print(f"✅ Augmented dataset saved with {len(new_df)} rows.")
