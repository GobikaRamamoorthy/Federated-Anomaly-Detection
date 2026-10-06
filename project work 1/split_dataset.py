import pandas as pd

# Load dataset
file_path = "all_faults_augmented.csv"
df = pd.read_csv(file_path)

# Split into 5 datasets based on Label
for label in df['Label'].unique():
    subset = df[df['Label'] == label]
    subset.to_csv(f"substation_{label}_aug.csv", index=False)
    print(f"Saved substation_{label}_aug.csv with {len(subset)} samples")

