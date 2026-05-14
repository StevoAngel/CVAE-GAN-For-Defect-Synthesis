"""
app.py
Interactive inference interface for the CVAE-GAN industrial defect generator.
"""

import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
from model import CVAE  # Import the architecture defined in model.py

# ==============================================================================
# 1. Environment Setup and Model Loading
# ==============================================================================

# In free HF Spaces, we use CPU. If a GPU is available, it will automatically switch to CUDA.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network parameters (Must match your training configuration)
LATENT_DIMS = 128
NUM_CLASSES = 2

# Instantiate the model
model = CVAE(latent_dims=LATENT_DIMS, num_classes=NUM_CLASSES)

# PATHS TO YOUR WEIGHTS (Make sure to upload them to your repository in these folders)
weights_path = "demo/checkpoints/cvae_gen.pth"
vector_path = "demo/checkpoints/vector_defect.pt"

# Load the model weights
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
else:
    print(f"WARNING: Weights file not found at {weights_path}")

# Load the directional defect vector
if os.path.exists(vector_path):
    vector_defect = torch.load(vector_path, map_location=device)
else:
    print(f"WARNING: Defect vector not found. Using a random vector as fallback.")
    vector_defect = torch.randn(1, LATENT_DIMS).to(device)

# Fixed labels (One-Hot Encoding for OK and Defect)
# Assuming 0 is OK and 1 is Defect based on your class_to_idx
label_ok = torch.tensor([[1.0, 0.0]]).to(device)
label_def = torch.tensor([[0.0, 1.0]]).to(device)

# ==============================================================================
# 2. Inference Function (Latent Arithmetic)
# ==============================================================================

def generate_defects(alpha, seed):
    """
    Generates two images: a healthy piece and a defective piece with porosity 
    based on the alpha multiplier.
    """
    # Set the seed so the jury can reproduce the exact same base geometry
    torch.manual_seed(int(seed))
    
    # 1. Generate the base healthy piece (z_base)
    z_base = torch.randn(1, LATENT_DIMS).to(device)
    
    with torch.no_grad():
        # Generate healthy image
        img_ok_tensor = model.decoder(z_base, label_ok)
        
        # 2. Latent Arithmetic: Inject the defect direction
        # Equation: z_modified = z_base + (alpha * vector_defect)
        z_modified = z_base + (alpha * vector_defect)
        
        # Generate defective image
        img_def_tensor = model.decoder(z_modified, label_def)
        
    # ==========================================================================
    # 3. Post-processing (From Tensor to PIL Image)
    # ==========================================================================
    
    def tensor_to_pil(tensor):
        # De-normalize: from [-1, 1] to [0, 1]
        tensor = tensor * 0.5 + 0.5
        # Convert to numpy, adjust dimensions and scale to [0, 255]
        ndarr = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ndarr = np.clip(ndarr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(ndarr)

    img_ok_pil = tensor_to_pil(img_ok_tensor)
    img_def_pil = tensor_to_pil(img_def_tensor)
    
    return img_ok_pil, img_def_pil

# ==============================================================================
# 3. User Interface with Gradio
# ==============================================================================

# Custom CSS for a cleaner, more academic look
custom_css = """
    .gradio-container { font-family: 'Inter', sans-serif; }
    .panel-header { font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏭 Síntesis de Porosidad Industrial mediante CVAE-GAN")
    gr.Markdown("""
    **Demostración Interactiva de Tesis de Maestría**
    
    Este espacio despliega un modelo generativo entrenado para tareas de **Computer Vision** en la industria metalúrgica. 
    Utilizando **Aritmética de Espacio Latente**, el modelo desacopla la geometría de la pieza metálica de la textura de la porosidad.
    
    * Ajusta el slider de **Intensidad de Defecto ($\alpha$)** para inyectar el vector direccional de anomalía en el espacio latente.
    * Cambia la **Semilla (Seed)** para explorar diferentes geometrías base de fundición.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div class='panel-header'>🎛️ Panel de Control</div>")
            alpha_slider = gr.Slider(minimum=0.0, maximum=3.0, value=1.0, step=0.1, 
                                     label="Intensidad de Defecto (α)")
            seed_input = gr.Number(value=42, label="Semilla (Geometría Base)", precision=0)
            generate_btn = gr.Button("Generar Inspección", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("<div class='panel-header'>📊 Resultados de Inferencia</div>")
            with gr.Row():
                out_ok = gr.Image(label="Pieza Sana (Referencia)", type="pil", interactive=False)
                out_def = gr.Image(label="Pieza Sintética (Con Defecto)", type="pil", interactive=False)
                
    # Button logic
    generate_btn.click(
        fn=generate_defects,
        inputs=[alpha_slider, seed_input],
        outputs=[out_ok, out_def]
    )

    # Generate an image when the page loads for the first time
    demo.load(
        fn=generate_defects,
        inputs=[alpha_slider, seed_input],
        outputs=[out_ok, out_def]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch()