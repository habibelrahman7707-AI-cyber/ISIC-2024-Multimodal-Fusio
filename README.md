Constructed a multimodal deep learning framework combining computer vision and tabular data to accurately classify and diagnose skin lesions based on the ISIC 2024 dataset.
Integrated a powerful EfficientNetV2 vision backbone (via timm) with a custom Multi-Layer Perceptron (MLP) head to simultaneously process high-resolution images and patient metadata (age, sex, anatomy).
Designed an advanced image augmentation pipeline using Albumentations (incorporating spatial transforms and color jittering) to prevent overfitting and handle severe class imbalances.
Optimized the training loop for speed and stability by leveraging PyTorch’s Automatic Mixed Precision (autocast), AdamW optimizer, and CosineAnnealingLR scheduler.
Restructured a complex monolithic script into a scalable, multi-file Python architecture (Data Loaders, Models, Train Loops), adhering to industry-standard ML engineering practices.
