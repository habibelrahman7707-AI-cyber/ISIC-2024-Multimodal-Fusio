# ISIC 2024 Skin Lesion Diagnosis - Multimodal Fusion Framework

A high-performance deep learning pipeline designed to solve the ISIC 2024 (SLICE-3D) challenge. This project implements a multimodal fusion architecture, combining a powerful vision backbone (`tf_efficientnetv2_s`) for image features with a Multi-Layer Perceptron (MLP) for tabular metadata (age, sex, anatomical site).

## 🚀 Architecture Overview
- **Vision Backbone**: Pretrained `EfficientNetV2-S` (via `timm`).
- **Metadata MLP**: A small neural network processing normalized metadata.
- **Fusion Head**: Concatenates image embeddings and metadata embeddings, passing them through a robust classifier block with Dropout and BatchNormalization.
- **Data Augmentation**: Advanced augmentation pipeline using `Albumentations` (RandomCrop, HueSaturationValue, ShiftScaleRotate).
- **Optimization**: Uses `AdamW`, `CosineAnnealingLR`, and Mixed Precision Training (`autocast`, `GradScaler`) for faster training.

## 📁 Project Structure
```
ISIC_2024_Project/
├── config.py             # Hyperparameters, Paths, and Seeds
├── data/
│   ├── dataset.py        # PyTorch Dataset for images and metadata
│   ├── preprocess.py     # CSV metadata cleaning, normalization, and encoding
│   └── transforms.py     # Albumentations augmentations
├── models/
│   └── fusion.py         # FusionModel Architecture
├── train.py              # Core training and validation loops
├── main.py               # Entry point to start the training pipeline
└── requirements.txt      # Project dependencies
```

## ⚙️ Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Preparation
Create a folder structure as follows and place your ISIC 2024 dataset inside:
```
data/ISIC2024/
├── metadata.csv
└── images/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

## 🎯 Usage
To begin training the fusion model, simply run:
```bash
python main.py
```
Outputs, including the best model weights (`best_isic2024_fusion.pth`) and the ROC validation curve (`roc_val.png`), will be saved to the `outputs/` directory.
