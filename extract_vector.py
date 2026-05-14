"""
extract_vector.py
Utility script to calculate and save the latent defect vector for inference.
This script DOES NOT train the model. It only extracts features using the pre-trained Encoder.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Add the 'demo' folder to the path so we can import your model architecture
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / 'demo'))
from model import CVAE

# ==============================================================================
# 1. Configuration and Paths
# ==============================================================================

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Architecture parameters (Must match training)
IMG_SIZE = 256
LATENT_DIMS = 128
NUM_CLASSES = 2
BATCH_SIZE = 64

# --- PATHS (Adjust dataset_path_root if your data is located elsewhere) ---
dataset_path_root = Path(r"C:\Users\estiv\OneDrive\Documents\Projects\3D_Defect_Models\data\raw\casting\casting_512x512")
weights_path = SCRIPT_DIR / "demo" / "checkpoints" / "cvae_gen.pth"
output_vector_path = SCRIPT_DIR / "demo" / "checkpoints" / "vector_defect.pt"

# ==============================================================================
# 2. Data Loading Preparation
# ==============================================================================

print("Preparing dataset and dataloader...")

transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3) # Z-Score normalization matching training
])

try:
    dataset = datasets.ImageFolder(root=str(dataset_path_root), transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Classes mapped: {dataset.class_to_idx}")
except FileNotFoundError:
    print(f"ERROR: Dataset not found at {dataset_path_root}. Please check the path.")
    sys.exit(1)

# ==============================================================================
# 3. Model Initialization
# ==============================================================================

print("Loading CVAE model...")
cvae = CVAE(latent_dims=LATENT_DIMS, num_classes=NUM_CLASSES).to(device)

if weights_path.exists():
    cvae.load_state_dict(torch.load(weights_path, map_location=device))
    cvae.eval() # STRICTLY set to evaluation mode to prevent any gradient updates
    print("Pre-trained weights loaded successfully.")
else:
    print(f"ERROR: Weights file not found at {weights_path}.")
    sys.exit(1)

# ==============================================================================
# 4. Extraction Logic
# ==============================================================================

@torch.no_grad() # Disable gradient calculation for memory efficiency and safety
def calculate_latent_centroids_dynamic(dataloader, dataset, encoder, device):
    print("Mapping the conditional Encoder's latent space...")
    
    # Extract the exact indices PyTorch assigned to your folders
    idx_def = dataset.class_to_idx.get('def_front', 0)
    idx_ok = dataset.class_to_idx.get('ok_front', 1)
    
    mu_def_list = []
    mu_ok_list = []

    for images, labels_idx in dataloader:
        images = images.to(device)
        labels_idx = labels_idx.to(device)
        
        # The Encoder needs encoded labels
        labels_encoded = F.one_hot(labels_idx, num_classes=NUM_CLASSES).float().to(device)
        
        # Pass images and labels through the Encoder
        mu, _ = encoder(images, labels_encoded)
        
        for i in range(len(labels_idx)):
            # Group vectors by class
            if labels_idx[i] == idx_def: 
                mu_def_list.append(mu[i].cpu().numpy())
            elif labels_idx[i] == idx_ok:
                mu_ok_list.append(mu[i].cpu().numpy())

    # Mathematical averages (Centroids)
    centroid_def = np.mean(mu_def_list, axis=0)
    centroid_ok = np.mean(mu_ok_list, axis=0)
    
    print(f"Extracted {len(mu_def_list)} defect vectors and {len(mu_ok_list)} OK vectors.")

    return torch.tensor(centroid_def).float().to(device), torch.tensor(centroid_ok).float().to(device)

# ==============================================================================
# 5. Execution and Saving
# ==============================================================================

def main():
    # Execute the extraction
    centroid_def, centroid_ok = calculate_latent_centroids_dynamic(dataloader, dataset, cvae.encoder, device)

    # Calculate the directional vector (Direction: from OK to Defect)
    vector_defect = centroid_def - centroid_ok
    print("Defect vector calculated successfully!")

    # Save the tensor locally in the checkpoints folder
    torch.save(vector_defect.cpu(), str(output_vector_path))
    print(f"SUCCESS: Vector saved to {output_vector_path}")
    print("You can now run your Gradio demo!")

if __name__ == "__main__":
    main()