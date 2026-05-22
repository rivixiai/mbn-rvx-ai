"""
MBN-RVX-AI: Magnetic Barkhausen Noise Signal Processing & Classification Pipeline
==================================================================================
This script provides the complete PyTorch implementation for assessing microstructural
degradation in ferromagnetic steels (such as 2.25Cr-1Mo) using Magnetic Barkhausen Noise.

Features:
- MBN 1D signal segmentation and STFT magnitude-to-dB conversion.
- Custom 2D Convolutional Neural Network (CNN) for binary degradation classification.
- Pipeline for training, evaluation, and inference.

For details, visit the project repository: https://github.com/rivixi/mbn-rvx-ai
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. Signal Preprocessing & Feature Extraction
# =====================================================================

def preprocess_mbn_signal(file_path, target_shape=(256, 256), duration=3.0, sr=100000):
    """
    Loads a 1D Magnetic Barkhausen Noise signal, segments/pads it to a target duration,
    calculates the STFT, converts it to dB scale, normalizes it, and resizes it to 
    the target 2D tensor shape.
    
    Parameters:
        file_path (str): Path to the WAV audio file containing the digitized MBN signal.
        target_shape (tuple): (height, width) for the output 2D spectrogram tensor.
        duration (float): Target segment duration in seconds.
        sr (int): Sampling rate of the digitized MBN signal (e.g., 100 kHz).
        
    Returns:
        torch.Tensor: Normalized 2D spectrogram tensor of shape [1, H, W].
    """
    # 1. Load the raw MBN 1D signal (high sampling rate)
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    
    # Pad or truncate signal to match exactly the required duration
    target_length = int(duration * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
        
    # 2. Compute Short-Time Fourier Transform (STFT) using a Hanning window
    # Window size and hop length are optimized for MBN frequency range
    n_fft = 1024
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    
    # 3. Convert amplitude to decibel (dB) scale to compress dynamic range
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # 4. Normalize to [0, 1] range for neural network input stability
    s_min, s_max = spectrogram_db.min(), spectrogram_db.max()
    if s_max - s_min > 1e-6:
        normalized = (spectrogram_db - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(spectrogram_db)
        
    # 5. Convert to PyTorch Tensor and add channel dimension (C, H, W)
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0) # [1, H, W]
    
    # Resize to target shape [1, 256, 256] using bilinear interpolation
    tensor = nn.functional.interpolate(
        tensor.unsqueeze(0), 
        size=target_shape, 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    return tensor

# =====================================================================
# 2. Convolutional Neural Network (CNN) Architecture
# =====================================================================

class MBNClassifier(nn.Module):
    """
    2D CNN model optimized for classifying MBN spectrograms.
    Categorizes the metal structure into:
    - 0: Baseline (Pristine / No service exposure)
    - 1: Degraded (Advanced creep / 120,000h service exposure)
    """
    def __init__(self):
        super(MBNClassifier, self).__init__()
        
        # Conv block 1: Extracts local spectral-temporal textures
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output size: [32, 128, 128]
        
        # Conv block 2: Extracts high-level frequency shift patterns
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output size: [64, 64, 64]
        
        # Fully connected classifier with Dropout regularization
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64 * 64, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) # Protects against overfitting on polar datasets
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid() # Outputs degradation probability P in [0, 1]
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

# =====================================================================
# 3. Model Training & Evaluation Template
# =====================================================================

class MBNDataset(Dataset):
    """Custom Dataset for loading preprocessed MBN spectrogram files."""
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        label = self.labels[idx]
        # Preprocess signal on the fly or load pre-extracted tensors
        tensor = preprocess_mbn_signal(file_path)
        return tensor, torch.tensor(label, dtype=torch.float32).unsqueeze(0)


def train_model(model, train_loader, val_loader, epochs=30, lr=1e-4, device='cpu'):
    """
    Trains the CNN model using Binary Cross-Entropy Loss and the Adam optimizer.
    Includes early stopping monitoring validation loss.
    """
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Binary classification accuracy threshold = 0.5
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model parameters
            torch.save(model.state_dict(), 'best_mbn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

# =====================================================================
# 4. Pipeline Inference Run Demonstration
# =====================================================================

def predict_degradation(file_path, model_path='best_mbn_model.pth', device='cpu'):
    """Runs inference on a single WAV signal to predict degradation probability."""
    # 1. Initialize and load model architecture
    model = MBNClassifier()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Extract 2D spectrogram tensor
    input_tensor = preprocess_mbn_signal(file_path).unsqueeze(0) # Add batch dimension [1, 1, H, W]
    input_tensor = input_tensor.to(device)
    
    # 3. Model Inference
    with torch.no_grad():
        probability = model(input_tensor).item()
        
    verdict = "DEGRADED (120,000h Creep Stage)" if probability >= 0.5 else "PRISTINE (Baseline)"
    return probability, verdict


if __name__ == "__main__":
    # Example usage:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running MBN-RVX-AI Pipeline on device: {device}")
    
    # 1. Instantiate the CNN Model
    model = MBNClassifier()
    print("Model Architecture successfully instantiated.")
    print(model)
    
    # To run training or inference, prepare .wav recordings and execute:
    # predict_degradation("path/to/pipeline_scan.wav", model_path="best_mbn_model.pth", device=device)
