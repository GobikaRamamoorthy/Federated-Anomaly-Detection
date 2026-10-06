import torch
import torch.nn as nn
import pickle
import numpy as np
from collections import OrderedDict

# ========== Define Autoencoder Architecture (Same as Local Models) ==========
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

# ========== Federated Averaging Function ==========
def federated_averaging(model_configs, equal_weights=False):
    """
    Aggregate multiple local model weights using FedAvg algorithm.
    
    Args:
        model_configs: List of dicts with keys:
            - 'model_path': Path to local model state_dict
            - 'num_samples': Number of training samples (for weighting)
            - 'name': Substation name
        equal_weights: If True, use equal weights instead of sample-based weights
    
    Returns:
        Aggregated state_dict
    """
    print("="*60)
    print("Federated Averaging - Aggregating Local Models")
    print("="*60)
    
    # Calculate weights based on number of samples
    if equal_weights:
        weights = [1.0 / len(model_configs)] * len(model_configs)
        print("Using equal weights for all models")
    else:
        total_samples = sum([config['num_samples'] for config in model_configs])
        weights = [config['num_samples'] / total_samples for config in model_configs]
        print("Using sample-proportional weights:")
    
    for i, config in enumerate(model_configs):
        print(f"  {config['name']:15s}: {config['num_samples']:6d} samples, weight = {weights[i]:.4f}")
    
    # Initialize global state
    global_state = OrderedDict()
    
    print("\nAggregating model parameters...")
    for i, config in enumerate(model_configs):
        print(f"  Loading {config['name']}...", end=" ")
        local_state = torch.load(config['model_path'])
        
        if i == 0:
            # Initialize global state with first model weighted
            for key in local_state.keys():
                global_state[key] = local_state[key].clone() * weights[i]
        else:
            # Aggregate remaining models
            for key in local_state.keys():
                global_state[key] += local_state[key] * weights[i]
        
        print("✓")
    
    print("\n✓ Global model aggregated successfully!")
    return global_state

# ========== Aggregate Local Thresholds ==========
def aggregate_thresholds(threshold_configs):
    """
    Aggregate local thresholds to create global threshold mapping.
    
    Args:
        threshold_configs: List of dicts with keys:
            - 'path': Path to threshold pickle file
            - 'label': Anomaly label (1-4)
            - 'name': Substation name
    
    Returns:
        Dictionary mapping labels to thresholds
    """
    print("\n" + "="*60)
    print("Aggregating Local Thresholds")
    print("="*60)
    
    global_thresholds = {}
    
    for config in threshold_configs:
        with open(config['path'], 'rb') as f:
            threshold_info = pickle.load(f)
        
        threshold = threshold_info['threshold']
        label = config['label']
        global_thresholds[label] = threshold
        
        print(f"  {config['name']:15s} (Label {label}): Threshold = {threshold:.6f}")
    
    print("\n✓ Global thresholds created!")
    return global_thresholds

# ========== Multi-Class Classification Function ==========
def classify_anomaly(sample, model, scaler, thresholds):
    """
    Classify anomaly type based on reconstruction errors.
    
    Args:
        sample: Input feature vector (numpy array)
        model: Trained global autoencoder
        scaler: StandardScaler from one of the local models
        thresholds: Dict of class-specific thresholds {label: threshold}
    
    Returns:
        predicted_label: Integer label (1=Sag, 2=Swell, 3=Transient, 4=Harmonic)
        reconstruction_error: Float value
    """
    model.eval()
    with torch.no_grad():
        # Normalize input
        sample_scaled = scaler.transform(sample.reshape(1, -1))
        sample_t = torch.tensor(sample_scaled, dtype=torch.float32)
        
        # Compute reconstruction error
        reconstructed = model(sample_t)
        error = torch.mean((reconstructed - sample_t)**2).item()
        
        # Find best matching class based on threshold proximity
        min_distance = float('inf')
        predicted_label = 1
        
        for label, threshold in thresholds.items():
            distance = abs(error - threshold)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label
        
        return predicted_label, error

# ========== Main Federated Learning Pipeline ==========
def create_global_model(input_dim=None):
    """
    Create global federated model from local models.
    
    Args:
    
        input_dim: Input dimension. If None, will be inferred from first model
    """
    
    # Configuration for local models
    # NOTE: Update 'num_samples' based on your actual training set sizes
    model_configs = [
        {
            'model_path': 'substation_1_best_autoencoder.pth',
            'num_samples': 3000,  # Replace with actual training size
            'name': 'Substation 1 (Sag)'
        },
        {
            'model_path': 'substation_2_best_autoencoder.pth',
            'num_samples': 3000,  # Replace with actual training size
            'name': 'Substation 2 (Swell)'
        },
        {
            'model_path': 'substation_3_best_autoencoder.pth',
            'num_samples':3000,  # Replace with actual training size
            'name': 'Substation 3 (Transient)'
        },
        {
            'model_path': 'substation_4_best_autoencoder.pth',
            'num_samples': 3000,  # Replace with actual training size
            'name': 'Substation 4 (Harmonic)'
        }
    ]
    
    # Configuration for thresholds
    threshold_configs = [
        {'path': 'threshold_substation_1.pkl', 'label': 1, 'name': 'Substation 1 (Sag)'},
        {'path': 'threshold_substation_2.pkl', 'label': 2, 'name': 'Substation 2 (Swell)'},
        {'path': 'threshold_substation_3.pkl', 'label': 3, 'name': 'Substation 3 (Transient)'},
        {'path': 'threshold_substation_4.pkl', 'label': 4, 'name': 'Substation 4 (Harmonic)'}
    ]
    
    # Infer input dimension if not provided
    if input_dim is None:
        first_model_state = torch.load(model_configs[0]['model_path'])
        input_dim = first_model_state['encoder.0.weight'].shape[1]
        print(f"Inferred input dimension: {input_dim}")
    
    # Apply Federated Averaging
    aggregated_state = federated_averaging(model_configs, equal_weights=False)
    
    # Create global model
    global_model = Autoencoder(input_dim, latent_dim=16)
    global_model.load_state_dict(aggregated_state)
    
    # Aggregate thresholds
    global_thresholds = aggregate_thresholds(threshold_configs)
    
    # Save global model
    torch.save(global_model.state_dict(), "global_federated_model.pth")
    print("\n✓ Saved: global_federated_model.pth")
    
    # Save global thresholds
    with open('global_thresholds.pkl', 'wb') as f:
        pickle.dump(global_thresholds, f)
    print("✓ Saved: global_thresholds.pkl")
    
    # Load one of the scalers (they should be similar since same feature space)
    with open('scaler_substation_1.pkl', 'rb') as f:
        global_scaler = pickle.load(f)
    
    with open('global_scaler.pkl', 'wb') as f:
        pickle.dump(global_scaler, f)
    print("✓ Saved: global_scaler.pkl")
    
    print("\n" + "="*60)
    print("Global Federated Model Creation Complete!")
    print("="*60)
    print("\nModel Summary:")
    print(f"  Architecture: Autoencoder with {input_dim}-dim input")
    print(f"  Latent dimension: 16")
    print(f"  Number of local models aggregated: {len(model_configs)}")
    print(f"  Anomaly classes: 4 (Sag, Swell, Transient, Harmonic)")
    
    return global_model, global_thresholds, global_scaler

# ========== Usage Example ==========
def inference_example(global_model, global_scaler, global_thresholds, test_sample):
    """
    Example of how to use the global model for inference.
    
    Args:
        global_model: Trained global autoencoder
        global_scaler: Feature scaler
        global_thresholds: Class-specific thresholds
        test_sample: Numpy array of features
    """
    label_map = {1: 'Sag', 2: 'Swell', 3: 'Transient', 4: 'Harmonic'}
    
    predicted_label, error = classify_anomaly(
        test_sample, global_model, global_scaler, global_thresholds
    )
    
    print(f"\nPrediction: {label_map[predicted_label]} (Label {predicted_label})")
    print(f"Reconstruction Error: {error:.6f}")
    print(f"Threshold for {label_map[predicted_label]}: {global_thresholds[predicted_label]:.6f}")

# ========== Main Execution ==========
if __name__ == "__main__":
    # Create global federated model
    global_model, global_thresholds, global_scaler = create_global_model()
    
    print("\n" + "="*60)
    print("Global Model Ready for Inference!")
    print("="*60)
    print("\nTo use the model:")
    print("  1. Load: global_federated_model.pth")
    print("  2. Load: global_scaler.pkl")
    print("  3. Load: global_thresholds.pkl")
    print("  4. Call: classify_anomaly(sample, model, scaler, thresholds)")
    print("\nLabel Mapping:")
    print("  1 = Sag")
    print("  2 = Swell")
    print("  3 = Transient")
    print("  4 = Harmonic")