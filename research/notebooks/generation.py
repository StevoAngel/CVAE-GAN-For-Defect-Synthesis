import os
import sys
import random
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

# Resolve repository root (folder that contains demo/model.py) from this script location.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    parent for parent in SCRIPT_PATH.parents if (parent / "demo" / "model.py").exists()
)
sys.path.insert(0, str(PROJECT_ROOT))

from demo.model import CVAE

# =====================================
# 1. Random seed to ensure reproducibility
# =====================================
def set_seed(seed=42):
    # 1. Native python seed:
    random.seed(seed)

    # 2. Evironment python seed:
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Numpy seed:
    np.random.seed(seed)
    
    # 4. PyTorch seed (CPU)
    torch.manual_seed(seed)
    
    # 5. PyTorch seed (GPU / CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Si usas múltiples GPUs
    
    # 6. CuDNN deterministic for stability in math operations:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) # Call seed function


# ==============================================================================
# 2. Paths and HW Configurations
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = 256
LATENT_DIMS = 128 
NUM_IMAGES = 1000  # Sample size of images to generate for evaluation
NUM_CLASSES = 2  # OK and Defect

# Model checkpoint path
model_checkpoint_path = PROJECT_ROOT / "demo" / "checkpoints" / "cvae_gen.pth"

# Output paths for evaluation sets
output_dir_ok = PROJECT_ROOT / "research" / "data" / "processed" / "generated_casting" / "cvae_ok"
output_dir_def = PROJECT_ROOT / "research" / "data" / "processed" / "generated_casting" / "cvae_def"
output_dir_ok.mkdir(parents=True, exist_ok=True)
output_dir_def.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 3. Load Model
# ==============================================================================
print("Loading CVAE-GAN Model Generator...")

try:
    model = CVAE(latent_dims=LATENT_DIMS, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(str(model_checkpoint_path), map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded successfully. Ready to generate images.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the checkpoint file exists and is compatible with the CVAE architecture.")

# For this case, only the decoder is necessary:
decoder = model.decoder
decoder.eval()


# ==============================================================================
# 4. Post-Processing Function (Tensor to Image)
# ==============================================================================
def tensor_to_image(tensor):
    "Converts a PyTorch tensor scaled at [-1, 1] to a PIL Image in [0, 255]"

    # Unnormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = tensor.clamp(0, 1)  # Ensure values are in [0, 1]

    # Reshape and convert to numpy array:
    numpy_img = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()  # From (C, H, W) to (H, W, C)
    numpy_img = (numpy_img * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    # Handle single-channel images (if C=1, convert to grayscale)
    if numpy_img.shape[2] == 1:
        return Image.fromarray(numpy_img[:, :, 0], mode='L')
    return Image.fromarray(numpy_img, mode='RGB')

# ==============================================================================
# 5. Image Generation Loop:
# ==============================================================================
print("Generating images...")

# Set a fixed seed for reproducibility of the base geometry:
base_seed = 42

# Fixed labels (One-Hot Encoding for OK and Defect)
# Assuming 0 is OK and 1 is Defect based on your class_to_idx
label_ok = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
label_def = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device)

with torch.no_grad():
    for i in tqdm(range(NUM_IMAGES), desc="Generating CVAE-GAN Images"):
        
        # 1. Sample standard Gaussian noise using a fixed progressive seed
        # This guarantees pairwise alignment: cvae_ok_001 will have the same base noise as cvae_def_001
        torch.manual_seed(base_seed + i)
        z = torch.randn(1, LATENT_DIMS, device=device)
        
        # 2. Execute forward pass through the Decoder explicitly
        fake_image_ok = decoder(z, label_ok)
        fake_image_def = decoder(z, label_def)
        
        # 3. Convert to PIL Images
        img_ok = tensor_to_image(fake_image_ok)
        img_def = tensor_to_image(fake_image_def)
        
        # 4. Resize to target resolution (if native output differs from 256x256)
        if img_ok.size != (TARGET_SIZE, TARGET_SIZE):
            img_ok = img_ok.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
            img_def = img_def.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
            
        # 5. Save to target directories
        img_ok.save(output_dir_ok / f"cvae_ok_{i:04d}.png")
        img_def.save(output_dir_def / f"cvae_def_{i:04d}.png")

print("CVAE-GAN image generation completed successfully!")