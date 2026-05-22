# MBN-RVX-AI: Magnetic Barkhausen Noise Signal Processing & Classification Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the complete PyTorch implementation of the signal processing and Deep Learning pipeline for assessing microstructural degradation in ferromagnetic steels (specifically **2.25Cr-1Mo** / ASTM A335 Grade P22) using Magnetic Barkhausen Noise (MBN).

The method provides a rapid, non-destructive testing (NDT) screening alternative to traditional manual replication (metallography) during planned plant outages/shutdowns.

## Directory Structure

```text
mbn-rvx-ai/
├── src/
│   └── mbn_ai_pipeline.py    # Main script (Signal preprocessing, STFT, CNN, Trainer, Predictor)
├── .gitignore                # Git exclusion file
├── README.md                 # Project documentation (this file)
└── requirements.txt          # Python dependencies
```

## Features

1. **Signal Preprocessing:**
   - 1D MBN signal segmentation into overlapping windows ($T=3.0\text{s}$, $50\%$ overlap).
   - Short-Time Fourier Transform (STFT) conversion with a Hanning window ($N_{\text{fft}}=1024$, $N_{\text{hop}}=512$).
   - Logarithmic scaling (amplitude to decibels $S_{\text{dB}}$) and Min-Max normalization.
   - Resizing to a fixed $256 \times 256$ grayscale spectrogram tensor via bilinear interpolation.
2. **CNN Architecture:**
   - Custom 2D CNN optimized for edge computing and physics-based acoustic/vibrational features.
   - Dual-block convolutional feature extractor (32 and 64 filters of size $3 \times 3$) with ReLU activations.
   - Max-pooling spatial reduction ($2 \times 2$) to decrease tensor size.
   - Sigmoid classifier yielding the final degradation probability $P \in [0, 1]$.
3. **Robust Training Framework:**
   - **Specimen-level data split** (strict split at the physical pipe level rather than segment level to prevent data leakage).
   - Early stopping based on validation loss, Dropout ($p=0.5$), and Adam optimizer with L2 weight decay.
   - Classification accuracy on unseen test specimens exceeds **92%** (AUC-ROC > 0.96).

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/rivixi/mbn-rvx-ai.git
cd mbn-rvx-ai
pip install -r requirements.txt
```

## How to Work with the Code

All code is contained within the `src/mbn_ai_pipeline.py` script. It can be imported into your Python scripts or notebooks.

### 1. Training the Model
To train the CNN model on your custom dataset of MBN measurements, organize your WAV files into directories and run:

```python
import torch
from torch.utils.data import DataLoader
from src.mbn_ai_pipeline import MBNClassifier, MBNDataset, train_model

# 1. Define paths and labels for your datasets
# (0 = Pristine/Baseline, 1 = Degraded/120k Hours)
train_files = ["data/train/pristine_1.wav", "data/train/degraded_1.wav"]
train_labels = [0, 1]

val_files = ["data/val/pristine_2.wav", "data/val/degraded_2.wav"]
val_labels = [0, 1]

# 2. Create PyTorch datasets and loaders
train_dataset = MBNDataset(train_files, train_labels)
val_dataset = MBNDataset(val_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. Instantiate model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MBNClassifier()

# 4. Run the training pipeline
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    lr=1e-4,
    device=device
)
```

### 2. Performing Inference on Single WAV File
To check a new MBN measurement WAV file for microstructural degradation:

```python
from src.mbn_ai_pipeline import predict_degradation

# Provide the path to the WAV file and the trained weights (.pth)
prob, verdict = predict_degradation(
    file_path="data/test/unlabeled_scan.wav",
    model_path="best_mbn_model.pth",
    device="cpu"
)

print(f"Degradation Probability: {prob:.4f}")
print(f"Verdict: {verdict}")
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- **Evgeny Ivanaiskiy, PhD** - Domain Expert
- **Alexander Ivanaiskiy, PhD** - Industrial AI Founder & Systems Architect
- **Sergey Shipilov** - AI Architecture Lead, Rivixi LLC
