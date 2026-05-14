# CVAE-GAN Industrial Defect Generator

This repository hosts the official implementation and demonstration of an industrial surface defect synthesizer. This project is a core component of a Master's Thesis focused on enhancing data availability for **Computer Vision** applications in the metal foundry industry.

## Overview

The synthesis of stochastic defects, such as porosity, remains a significant challenge in the field of **Deep Learning (DL)** applied to industrial inspection. This solution utilizes a **Conditional Variational Autoencoder (CVAE)** integrated with a **Generative Adversarial Network (GAN)**—collectively referred to as **CVAE-GAN**—to model the complex distribution of metallic textures.

By disentangling the latent space, this model allows for the controlled generation of "good" (defect-free) and "bad" (defective) samples from the same underlying geometry, providing a robust tool for data augmentation in **Computer Vision** pipelines.

## Key Features

* **Controlled Synthesis:** Fine-grained control over defect intensity via latent vector manipulation.
* **Structural Consistency:** Preservation of the metallic piece's geometry and lighting conditions through the CVAE-GAN architecture.
* **Dual Output:** Simultaneous generation of healthy and defective counterparts for comparative analysis.
* **Research-Oriented:** Designed specifically to address the "data scarcity" problem in industrial quality control.

## Technical Foundation

The architecture is based on the principles of variational inference and adversarial training, drawing from the following foundational research:
* **# CVAE-GAN Industrial Defect Generator

This repository hosts the official implementation and demonstration of an industrial surface defect synthesizer. This project is a core component of a Master's Thesis focused on enhancing data availability for **Computer Vision** applications in the metal foundry industry.

## Overview

The synthesis of stochastic defects, such as porosity, remains a significant challenge in the field of **Deep Learning (DL)** applied to industrial inspection. This solution utilizes a **Conditional Variational Autoencoder (CVAE)** integrated with a **Generative Adversarial Network (GAN)**—collectively referred to as **CVAE-GAN**—to model the complex distribution of metallic textures.

By disentangling the latent space, this model allows for the controlled generation of "good" (defect-free) and "bad" (defective) samples from the same underlying geometry, providing a robust tool for data augmentation in **Computer Vision** pipelines.

## Key Features

* **Controlled Synthesis:** Fine-grained control over defect intensity via latent vector manipulation.
* **Structural Consistency:** Preservation of the metallic piece's geometry and lighting conditions through the CVAE-GAN architecture.
* **Dual Output:** Simultaneous generation of healthy and defective counterparts for comparative analysis.
* **Research-Oriented:** Designed specifically to address the "data scarcity" problem in industrial quality control.

## Technical Foundation

The architecture is based on the principles of variational inference and adversarial training, drawing from the following foundational research:
* **CVAE-GAN: Fine-Grained Image Generation through Adversarial Training** (Bao et al., 2017).
* **Deep Residual Learning for Image Recognition** (He et al., 2016).
* **Auto-Encoding Variational Bayes** (Kingma & Welling, 2013).
* **Conditional Image Synthesis with Auxiliary Classifier GANs** (Odena et al., 2017).
* **Learning Structured Output Representation using Deep Conditional Generative Models** (Sohn et al., 2015).

## Hardware & Performance

The model was optimized for efficient inference, allowing it to run on standard hardware while maintaining the high fidelity required for **Computer Vision** defect detection tasks.
