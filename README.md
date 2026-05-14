# Deepfake Image Detection Using Hybrid Vision Transformers and CNNs

This repository contains the implementation and research findings for a robust deepfake detection system that leverages hybrid deep learning architectures. By combining the hierarchical feature extraction of Convolutional Neural Networks (CNNs) with the global contextual understanding of Vision Transformers (ViTs), our models achieve superior performance in identifying AI-generated facial manipulations.

## 🚀 Key Features
- **Hybrid Architectures**: Integration of **SWIN Transformer** with state-of-the-art backbones including **ConvNeXt**, **DenseNet**, **MobileNetV3**, and **CLIP**.
- **Multi-Branch Analysis**:
  - **Spatial Branch**: Captures micro-textural anomalies and long-range spatial dependencies.
  - **Frequency Branch**: Utilizes FFT-based analysis to detect subtle spectral artifacts introduced during generation.
- **Attention Fusion**: A dynamic attention mechanism to weigh contributions from different architectural branches for optimal classification.
- **Unified Dataset**: Trained on a comprehensive blend of **DFDC**, **FaceForensics++ (FF++)**, **Celeb-DF (v2)**, and **140k Real and Fake Faces**.

## 🛠️ Implementation Details
- **Preprocessing**: Facial extraction using MTCNN with images standardized to $224 \times 224$ pixels.
- **Optimizations**: Automatic Mixed Precision (AMP), OneCycleLR/Cosine Annealing schedulers, and early stopping to prevent overfitting.
- **Environment**: Developed for both high-performance and resource-constrained (MobileNet-based) deployments.
