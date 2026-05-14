# Deepfake Image Detection Using Hybrid Vision Transformers and CNNs

This repository contains the implementation and research findings for a robust deepfake detection system that leverages hybrid deep learning architectures. By combining the hierarchical feature extraction of Convolutional Neural Networks (CNNs) with the global contextual understanding of Vision Transformers (ViTs), our models achieve superior performance in identifying AI-generated facial manipulations.

## 🚀 Key Features
- **Hybrid Architectures**: Integration of **SWIN Transformer** with state-of-the-art backbones including **ConvNeXt**, **DenseNet**, **MobileNetV3**, and **CLIP**.
- **Multi-Branch Analysis**:
  - **Spatial Branch**: Captures micro-textural anomalies and long-range spatial dependencies.
  - **Frequency Branch**: Utilizes FFT-based analysis to detect subtle spectral artifacts introduced during generation.
- **Attention Fusion**: A dynamic attention mechanism to weigh contributions from different architectural branches for optimal classification.
- **Unified Dataset**: Trained on a comprehensive blend of **DFDC**, **FaceForensics++ (FF++)**, **Celeb-DF (v2)**, and **140k Real and Fake Faces**.

## 📊 Performance Summary
Our hybrid models consistently outperform standalone counterparts across all key metrics:

| Model | Test Accuracy | F1 Score | Recall | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: |
| **ConvNeXt-SWIN Hybrid** | **93.47%** | **93.01%** | 93.55% | **0.9831** |
| **CLIP-SWIN Hybrid** | 91.11% | 90.84% | **94.99%** | 0.9741 |
| **MobileNet-SWIN Hybrid** | 92.28% | 92.71% | 91.63% | 0.9786 |
| **DenseNet-SWIN Hybrid** | 91.36% | 90.75% | 91.30% | 0.9737 |

## 🛠️ Implementation Details
- **Preprocessing**: Facial extraction using MTCNN with images standardized to $224 \times 224$ pixels.
- **Optimizations**: Automatic Mixed Precision (AMP), OneCycleLR/Cosine Annealing schedulers, and early stopping to prevent overfitting.
- **Environment**: Developed for both high-performance and resource-constrained (MobileNet-based) deployments.
