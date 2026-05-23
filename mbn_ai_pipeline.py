"""
MBN-RVX-AI: Magnetic Barkhausen Noise Signal Processing & Classification Pipeline
==================================================================================
This script provides the complete PyTorch implementation for assessing microstructural
degradation in ferromagnetic steels (such as 2.25Cr-1Mo) using Magnetic Barkhausen Noise.

Features:
- MBN 1D signal segmentation using an overlapping sliding window.
- Offline preprocessing to .pt files for accelerated GPU training.
- Custom 2D Convolutional Neural Network (CNN) for binary structural classification.
- Comprehensive metrics evaluation (Accuracy, F1, AUC-ROC, Confusion Matrix).
- Robust batched inference with multi-segment averaging for field testing.

Classes:
- 0: Acceptable state (no critical microstructural defects)
- 1: Degraded state (unacceptable defects: creep voids, advanced spheroidization)

For details, visit the project repository: https://github.com/rivixi/mbn-rvx-ai
"""

import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

# =====================================================================
# 0. Reproducibility Configuration
# =====================================================================

def seed_everything(seed=42):
    """Fixes random seeds across libraries to ensure reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# =====================================================================
# 1. Signal Preprocessing & Segment-wise Feature Extraction
# =====================================================================

def extract_spectrogram(y, sr, target_shape=(256, 256)):
    """
    Computes the STFT spectrogram of a 1D signal chunk, translates it to dB,
    normalizes it to [0, 1] range, and resizes it to the target tensor shape.
    """
    # 1. Compute Short-Time Fourier Transform (STFT)
    n_fft = 1024
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    
    # 2. Convert amplitude to decibel (dB) scale
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # 3. Normalize to [0, 1] range
    s_min, s_max = spectrogram_db.min(), spectrogram_db.max()
    if s_max - s_min > 1e-6:
        normalized = (spectrogram_db - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(spectrogram_db)
        
    # 4. Convert to PyTorch Tensor [C, H, W]
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0) # [1, H, W]
    
    # 5. Bilinearly interpolate to the target shape [1, 256, 256]
    tensor = nn.functional.interpolate(
        tensor.unsqueeze(0), 
        size=target_shape, 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    return tensor


def segment_and_preprocess_signal(file_path, duration=3.0, overlap=0.5, sr=100000, target_shape=(256, 256)):
    """
    Loads a full 1D MBN signal, segments it using a sliding window with overlap,
    and converts each segment into a 2D spectrogram tensor.
    """
    # Load raw MBN 1D signal
    y, sr = librosa.load(file_path, sr=sr)
    
    segment_len = int(duration * sr)
    step_len = int(segment_len * (1 - overlap))
    
    tensors = []
    
    # If the file is shorter than target duration, pad it and extract single segment
    if len(y) < segment_len:
        y_padded = np.pad(y, (0, segment_len - len(y)), mode='constant')
        tensors.append(extract_spectrogram(y_padded, sr, target_shape))
        return tensors
        
    # Slice the signal with the overlapping sliding window
    for start in range(0, len(y) - segment_len + 1, step_len):
        chunk = y[start : start + segment_len]
        tensors.append(extract_spectrogram(chunk, sr, target_shape))
        
    return tensors


def preprocess_and_save_dataset(wav_files, labels, output_dir, duration=3.0, overlap=0.5, sr=100000, target_shape=(256, 256)):
    """
    Preprocesses raw WAV files offline by slicing them and saving 2D spectrograms
    as separate .pt files. This accelerates training loops by eliminating on-the-fly STFT computation.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    saved_labels = []
    
    print(f"Starting offline preprocessing of {len(wav_files)} files into: {output_dir}")
    
    for idx, (file_path, label) in enumerate(zip(wav_files, labels)):
        try:
            tensors = segment_and_preprocess_signal(file_path, duration=duration, overlap=overlap, sr=sr, target_shape=target_shape)
            for s_idx, tensor in enumerate(tensors):
                filename = f"sample_{idx}_seg_{s_idx}.pt"
                save_path = os.path.join(output_dir, filename)
                
                # Save as a pre-packaged tuple (spectrogram_tensor, label_tensor)
                label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
                torch.save((tensor, label_tensor), save_path)
                
                saved_paths.append(save_path)
                saved_labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"Preprocessing complete. Saved {len(saved_paths)} segment files.")
    return saved_paths, saved_labels

# =====================================================================
# 2. Convolutional Neural Network (CNN) Architecture
# =====================================================================

class MBNClassifier(nn.Module):
    """
    2D CNN model optimized for classifying MBN spectrograms.
    Outputs the probability P that the metal contains unacceptable structural degradation.
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
        # Input features: 64 channels * 64 height * 64 width = 262,144 features
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
# 3. Model Training & Scientific Evaluation
# =====================================================================

class MBNDataset(Dataset):
    """Custom Dataset for loading pre-extracted .pt files or on-the-fly .wav files."""
    def __init__(self, file_list, labels=None, is_preprocessed=True):
        self.file_list = file_list
        self.labels = labels
        self.is_preprocessed = is_preprocessed

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        if self.is_preprocessed:
            # Load preprocessed tensor & label directly from .pt file
            return torch.load(file_path)
        else:
            # On-the-fly processing (fallback for small datasets or single checks)
            label = self.labels[idx]
            tensors = segment_and_preprocess_signal(file_path)
            tensor = tensors[0] if len(tensors) > 0 else torch.zeros((1, 256, 256))
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
                
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best parameters
            torch.save(model.state_dict(), 'best_mbn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


def evaluate_model(model, data_loader, device='cpu'):
    """
    Runs model evaluation over a validation/test dataset and calculates key metrics:
    Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix.
    """
    model.eval()
    model.to(device)
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_outputs = np.array(all_outputs).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Binary predictions using 0.5 threshold
    preds = (all_outputs >= 0.5).astype(float)
    
    # Metrics calculations
    accuracy = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_outputs)
    cm = confusion_matrix(all_labels, preds)
    
    print("\n==================== Model Evaluation Metrics ====================")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall (Sens.):  {recall:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"AUC-ROC Score:   {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("==================================================================")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }

# =====================================================================
# 4. Optimized Inference Run
# =====================================================================

def predict_degradation(file_path, model=None, model_path='best_mbn_model.pth', device='cpu'):
    """
    Runs inference on a single WAV signal to predict degradation probability.
    Accepts either an active pre-loaded model instance (optimized for loops) or a model path.
    """
    # 1. Initialize and load model if not provided
    if model is None:
        model = MBNClassifier()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Extract MBN segments using overlapping sliding window
    tensors = segment_and_preprocess_signal(file_path, sr=100000)
    if len(tensors) == 0:
        return 0.0, "UNKNOWN (Signal too short)"
        
    # Stack all segments into a batch for single-pass inference [N, 1, 256, 256]
    input_batch = torch.stack(tensors).to(device)
    
    # 3. Model Inference
    with torch.no_grad():
        probabilities = model(input_batch).cpu().numpy().flatten()
        
    # Average the probabilities across all window segments for a robust, high-fidelity verdict
    mean_probability = float(np.mean(probabilities))
    
    verdict = "DEGRADED (Unacceptable structural defects detected)" if mean_probability >= 0.5 else "ACCEPTABLE (Structure within safe limits)"
    return mean_probability, verdict


if __name__ == "__main__":
    # Fix random seed for reproducibility
    seed_everything(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running MBN-RVX-AI Pipeline on device: {device}")
    
    # 1. Instantiate the CNN Model
    model = MBNClassifier()
    print("Model Architecture successfully instantiated:")
    print(model)
    
    # To run training:
    # 1. Prepare list of WAV files: wav_paths = [...] and labels = [0, 1, 0, ...]
    # 2. Preprocess: pt_paths, _ = preprocess_and_save_dataset(wav_paths, labels, "data/processed")
    # 3. Split pt_paths into train/val subsets
    # 4. Load dataset: train_dataset = MBNDataset(train_pt_paths, is_preprocessed=True)
    # 5. Run: train_model(model, DataLoader(train_dataset, batch_size=16, shuffle=True), val_loader, device=device)
