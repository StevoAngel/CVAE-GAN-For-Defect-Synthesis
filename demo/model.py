"""
model.py
Architecture definitions for the CVAE-GAN Industrial Defect Generator.
This file contains the classes required to instantiate the model before 
loading the pre-trained weights (.pth) for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==============================================================================
# 1. Residual Block
# ==============================================================================

class ResidualBlock(nn.Module):
    """ 
    Residual block that keeps resolution and channels intact. 
    Its sole function is to process and refine high-frequency textures 
    (like porosity) without losing the geometric structure.
    """
    def __init__(self, channels):
        super().__init__()
        # First convolution (maintains size due to padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x # Save the original input for the skip connection
        
        # Pass through the main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add the skip connection before the final activation
        out += residual 
        return F.relu(out)

# ==============================================================================
# 2. Encoder (GoogLeNet based)
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, latent_dims, num_classes=2):
        super().__init__()
        
        # Load the base GoogLeNet (without auxiliary classifiers nor weights)
        googlenet = models.googlenet(pretrained=False, aux_logits=False)
        
        # Extract all layers EXCEPT the last fully connected (FC) layer
        self.feature_extractor = nn.Sequential(*list(googlenet.children())[:-1]) # 1024-dimensional feature vector
        
        # Fusion: 1024 image features only. The saved checkpoint was trained
        # with an unconditional encoder, so we keep the label argument for API
        # compatibility but do not concatenate it here.
        self.fc_fusion = nn.Linear(1024, 512)
        
        # Final mapping to the latent space z
        self.fc_mu = nn.Linear(512, latent_dims)
        self.fc_log_var = nn.Linear(512, latent_dims)

    def forward(self, x, label):
        # The input image is processed independently through GoogLeNet
        features = self.feature_extractor(x)    # Output: [batch, 1024, 1, 1]
        features = torch.flatten(features, 1)   # Output: [batch, 1024]

        # The encoder uses only image features; the label is kept in the
        # signature so existing call sites do not need to change.
        fused = F.relu(self.fc_fusion(features))
        
        mu = self.fc_mu(fused)
        log_var = self.fc_log_var(fused)
        
        return mu, log_var

# ==============================================================================
# 3. Decoder (Generator)
# ==============================================================================

class Decoder(nn.Module):
    def __init__(self, latent_dims, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Label Embedding: Increases the dimensionality of the conditional label
        self.label_emb = nn.Linear(num_classes, 64)
        
        # --- 2 Fully Connected (FC) Layers ---
        self.fc1 = nn.Linear(latent_dims + 64, 1024)
        self.fc2 = nn.Linear(1024, 512 * 4 * 4) 
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 4, 4))
        
        # --- 6 Deconvolution Layers (2-by-2 Upsampling) ---
        
        # SPATIAL INJECTION: Layer 1
        self.deconv1 = nn.ConvTranspose2d(512 + num_classes, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.res1 = ResidualBlock(256)
        
        # Layer 2
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.res2 = ResidualBlock(256)
        
        # Layer 3
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.res3 = ResidualBlock(128)
        
        # Layer 4
        self.deconv4 = nn.ConvTranspose2d(128, 92, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn4 = nn.BatchNorm2d(92)
        self.res4 = ResidualBlock(92)
        
        # Layer 5
        self.deconv5 = nn.ConvTranspose2d(92, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.res5 = ResidualBlock(64)
        
        # Layer 6: Final Image Output
        self.deconv6 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z, label):
        # 1. Embed the conditional label and concatenate it with the latent vector (z)
        label_embedded = F.relu(self.label_emb(label))
        x = torch.cat((z, label_embedded), dim=1)
        
        # Pass through the FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.unflatten(x) 
        
        # 2. DEEP SPATIAL INJECTION
        label_spatial = label.unsqueeze(2).unsqueeze(3)
        label_spatial = label_spatial.expand(-1, -1, x.size(2), x.size(3))
        
        # Concatenate the feature map with the label spatial map
        x = torch.cat((x, label_spatial), dim=1) 
        
        # Pass through Deconvolutions with Batch Normalization and ResNet blocks
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.res1(x)

        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.res2(x)

        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.res3(x)

        x = F.relu(self.bn4(self.deconv4(x)))
        x = self.res4(x)

        x = F.relu(self.bn5(self.deconv5(x)))
        x = self.res5(x)
        
        # The final layer uses Tanh for images in range [-1, 1], without BatchNorm
        x = torch.tanh(self.deconv6(x)) 
        
        return x

# ==============================================================================
# 4. CVAE (Master Module)
# ==============================================================================

class CVAE(nn.Module):
    def __init__(self, latent_dims=128, num_classes=2):
        super().__init__()
        self.encoder = Encoder(latent_dims, num_classes)
        self.decoder = Decoder(latent_dims, num_classes)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  
        eps = torch.randn_like(std)     
        return mu + eps * std
    
    def forward(self, x, label):
        mu, log_var = self.encoder(x, label)
        z = self.reparametrize(mu, log_var)
        return self.decoder(z, label), mu, log_var

# ==============================================================================
# 5. Discriminator (Optional for Inference, but required for loading full states)
# ==============================================================================

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.rf_classifier = nn.Conv2d(ndf * 16, 1, 4, stride=2, padding=1, bias=False)
            
    def forward(self, x):
        features = self.feature_extractor(x)
        output_rf = self.rf_classifier(features).view(x.size(0), -1).mean(1) 
        return output_rf, features