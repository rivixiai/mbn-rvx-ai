# MBN-RVX-AI: Magnetic Barkhausen Noise Signal Processing & Classification Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the complete PyTorch implementation of the signal processing and Deep Learning pipeline for assessing microstructural degradation in ferromagnetic steels (specifically **2.25Cr-1Mo** / ASTM A335 Grade P22) using Magnetic Barkhausen Noise (MBN).

The method provides a rapid, non-destructive testing (NDT) screening alternative to traditional manual replication (metallography) during planned plant outages/shutdowns, detecting unacceptable structural defects (creep voids, graphitization, spheroidization) in minutes.

## Directory Structure

```text
mbn-rvx-ai/
├── src/
│   └── mbn_ai_pipeline.py    # Main script (Preprocessing, STFT, CNN, Trainer, Evaluator, Predictor)
├── .gitignore                # Git exclusion file
├── README.md                 # Project documentation (this file)
└── requirements.txt          # Python dependencies
```

## Features

1. **Signal Preprocessing & Windowing:**
   - 1D MBN signal segmentation into overlapping sliding windows ($T=3.0\text{s}$, $50\%$ overlap) to support long field measurements.
   - Short-Time Fourier Transform (STFT) conversion with a Hanning window ($N_{\text{fft}}=1024$, $N_{\text{hop}}=512$).
   - Logarithmic scaling (amplitude to decibels $S_{\text{dB}}$) and Min-Max normalization.
   - Resizing to a fixed $256 \times 256$ grayscale spectrogram tensor.
2. **Accelerated Dataset Preparation:**
   - Offline preprocessing utility (`preprocess_and_save_dataset`) that converts raw WAV data into compact `.pt` PyTorch tensors.
   - Speeds up model training on GPUs by up to 20x by avoiding on-the-fly STFT computation.
3. **CNN Architecture & Reproducibility:**
   - Custom 2D CNN optimized for edge computing and physics-based acoustic/vibrational features.
   - Sigmoid classifier yielding the final degradation probability $P \in [0, 1]$.
   - Global reproducibility setup (`seed_everything`) fixing NumPy, Python random, and PyTorch seeds.
4. **Validation and Evaluation Metrics:**
   - Full evaluation loop calculating Classification Accuracy, F1-Score, Precision, Recall, Confusion Matrix, and AUC-ROC score (using `scikit-learn`).
5. **Robust Field Inference:**
   - Multi-segment averaging during single-file checks. Slices long field recordings into overlapping segments, runs inference on each, and averages the scores for a high-fidelity prediction.

## Classification Scheme
- **Class 0 (Acceptable state):** Structural state is within safe operating limits, showing no critical microstructural defects.
- **Class 1 (Degraded state):** Unacceptable structural degradation detected, indicating critical microstructural defects (creep voids, advanced spheroidization, microcracks).

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

For fast training, first preprocess the raw WAV files into `.pt` files, then run the trainer:

```python
import torch
from torch.utils.data import DataLoader
from src.mbn_ai_pipeline import MBNClassifier, MBNDataset, train_model, preprocess_and_save_dataset, seed_everything

# 1. Fix seed for reproducibility
seed_everything(42)

# 2. Define raw WAV files and labels (0 = Acceptable, 1 = Degraded)
wav_files = ["data/raw/pristine_bend.wav", "data/raw/damaged_bend.wav"]
labels = [0, 1]

# 3. Preprocess and save to disk
pt_paths, pt_labels = preprocess_and_save_dataset(wav_files, labels, output_dir="data/processed")

# 4. Load dataset (loads preprocessed .pt files)
train_dataset = MBNDataset(pt_paths, is_preprocessed=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 5. Instantiate model and run training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MBNClassifier()

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=train_loader,  # Replace with actual val_loader in real setups
    epochs=30,
    lr=1e-4,
    device=device
)
```

### 2. Evaluating the Model (Metrics)

To calculate accuracy, AUC-ROC, F1-Score, and the Confusion Matrix on your test dataset:

```python
from src.mbn_ai_pipeline import evaluate_model
# Loader containing your test/val data
evaluate_model(model, train_loader, device=device)
```

### 3. Performing Inference on Single WAV File
To check a new MBN measurement WAV file for microstructural degradation:

```python
from src.mbn_ai_pipeline import predict_degradation, MBNClassifier

# Load model once
model = MBNClassifier()
model.load_state_dict(torch.load("best_mbn_model.pth"))

# Run prediction
prob, verdict = predict_degradation(
    file_path="data/test/field_scan.wav",
    model=model,
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
