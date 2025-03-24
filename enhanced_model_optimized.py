#!/usr/bin/env python
"""
Piano Fingering Prediction - Enhanced Optimized Model
Combines CNN, BiLSTM, and multi-head attention architecture with successful data processing pipeline
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from models import CNNBiLSTMAttention, AttentionMechanism

# Class-balanced loss function for handling imbalanced classes
class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=0.9999, gamma=2.0, samples_per_class=None, weight=None, ignore_index=-100):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.samples_per_class = samples_per_class
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Apply class balanced weighting
        if self.samples_per_class is not None and self.weight is None:
            effective_num = 1.0 - torch.pow(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / effective_num
            weights = weights / torch.sum(weights) * len(self.samples_per_class)
            self.weight = weights.to(logits.device)

        # Create one-hot encoding for targets
        one_hot_targets = F.one_hot(targets, num_classes=logits.size(1)).float()
        
        # Focal loss component (works better for imbalanced data)
        probs = F.softmax(logits, dim=1)
        probs_t = torch.sum(probs * one_hot_targets, dim=1)
        focal_weight = torch.pow(1 - probs_t, self.gamma)
        
        # Weight the loss by the focal weight and class weight
        ce_loss = F.cross_entropy(
            logits, targets, 
            weight=self.weight, 
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Apply focal weighting
        loss = focal_weight * ce_loss
        
        # Ignore certain indices (like padding)
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index)
            loss = loss[valid_mask]
            
        return loss.mean()

# Enhanced fingering model with CNN, BiLSTM, and attention
class EnhancedFingeringModel(nn.Module):
    def __init__(self, input_size, seq_length=10, hidden_size=128, dropout=0.3, num_classes=10):
        super(EnhancedFingeringModel, self).__init__()
        
        # Input normalization for stability
        self.input_norm = nn.LayerNorm(input_size)
        
        # CNN feature extraction - capture local patterns
        self.cnn_layers = nn.Sequential(
            # First convolutional block
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),  # GELU activation for better gradient flow
            nn.Dropout(dropout/2),
            
            # Second convolutional block with residual connection
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # Residual connection adapter
        self.residual_adapter = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        
        # BiLSTM layers for sequential modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        # Multi-head attention mechanism
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size*2,  # BiLSTM output size
            num_heads=4,
            dropout=dropout
        )
        
        # Output classifier with skip connection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Store attention weights for analysis
        self.last_attention_weights = None
        
    def forward(self, x, hand_indices=None):
        # x shape: [batch_size, seq_len, features]
        batch_size, seq_len, features = x.size()
        
        # Normalize input for stable training
        x = self.input_norm(x)
        
        # CNN feature extraction
        x_cnn = x.transpose(1, 2)  # [batch, features, seq_len]
        x_cnn_out = self.cnn_layers(x_cnn)
        
        # Add residual connection
        residual = self.residual_adapter(x_cnn)
        x_cnn_out = x_cnn_out + residual
        
        x_cnn_out = x_cnn_out.transpose(1, 2)  # [batch, seq_len, hidden]
        
        # Process hand-specific feature if provided
        if hand_indices is not None:
            # Get hand information from the specified index
            if isinstance(hand_indices, int):
                hand_feature = x[:, -1, hand_indices].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
                # Adjust channel importance based on hand (simple attention)
                hand_weight = torch.sigmoid(hand_feature)
                # Broadcast weight to match dimensions
                hand_weight = hand_weight.expand(-1, x_cnn_out.size(1), 1)
                # Apply hand-specific weighting
                x_cnn_out = x_cnn_out * (1.0 + 0.5 * hand_weight)
        
        # BiLSTM processing
        lstm_out, _ = self.lstm(x_cnn_out)  # [batch, seq, hidden*2]
        
        # Multi-head attention
        # Reshape for attention: [seq_len, batch_size, hidden*2]
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, attn_weights = self.mha(
            lstm_out_transposed, 
            lstm_out_transposed, 
            lstm_out_transposed
        )
        # Store attention weights for visualization
        self.last_attention_weights = attn_weights.detach()
        
        # Back to [batch, seq, hidden*2]
        attn_out = attn_out.transpose(0, 1)
        
        # Take the last sequence position
        context = attn_out[:, -1, :]
        
        # Classification
        output = self.classifier(context)
        
        return output

# Compute class weights for balanced training
def compute_class_weights(y_train):
    class_counts = np.bincount(y_train[y_train < 10])  # Exclude padding if any
    total_samples = np.sum(class_counts)
    n_classes = len(class_counts)
    
    # Inverse frequency weighting with smoothing
    weights = total_samples / (n_classes * class_counts)
    
    # Apply logarithmic scaling to prevent extreme weights
    weights = 1 + np.log(weights)
    
    # Normalize weights
    weights = weights / np.sum(weights) * n_classes
    
    return torch.tensor(weights, dtype=torch.float32)

# Load data and prepare for training
def load_data(args):
    """Load preprocessed data files"""
    print(f"Loading data from: {args.data_dir}")
    
    # Load X and y arrays
    try:
        X_train = np.load(os.path.join(args.data_dir, 'X_train_aug.npy'))
        y_train = np.load(os.path.join(args.data_dir, 'y_train_aug.npy'))
        X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
        
        print(f"Data loaded successfully.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
        # Check for class distribution
        class_counts = np.bincount(y_train, minlength=10)
        for i, count in enumerate(class_counts):
            percentage = (count / len(y_train)) * 100
            print(f"Class {i}: {count} samples ({percentage:.2f}%)")
        
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Dataset class for sequence data
class SequenceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Training function for a single epoch
def train_epoch(model, loader, criterion, optimizer, device, clip_value=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    for X_batch, y_batch in tqdm(loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch, hand_indices=2)  # Assuming index 2 is hand feature
        
        # Only compute loss on labeled samples (ignore padding)
        mask = y_batch < 10  # Exclude padding class (10)
        if mask.sum() > 0:
            # Calculate loss
            loss = criterion(outputs[mask], y_batch[mask])
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                
            optimizer.step()
            
            total_loss += loss.item() * mask.sum().item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs[mask], 1)
            total += mask.sum().item()
            correct += (predicted == y_batch[mask]).sum().item()
            
            # Store predictions and targets for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch[mask].cpu().numpy())
    
    # Calculate epoch statistics
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy, all_predictions, all_targets

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch, hand_indices=2)
            
            # Only compute metrics on labeled samples
            mask = y_batch < 10
            if mask.sum() > 0:
                loss = criterion(outputs[mask], y_batch[mask])
                total_loss += loss.item() * mask.sum().item()
                
                _, predicted = torch.max(outputs[mask], 1)
                total += mask.sum().item()
                correct += (predicted == y_batch[mask]).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch[mask].cpu().numpy())
    
    # Calculate evaluation statistics
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-class accuracy
    conf_matrix = confusion_matrix(all_targets, all_predictions, labels=range(10))
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    return avg_loss, accuracy, class_acc, conf_matrix, all_predictions, all_targets

# Visualize training results
def plot_training_results(history, save_path='results_optimized.png'):
    plt.figure(figsize=(15, 10))
    
    # Plot training & validation accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot per-class accuracy
    plt.subplot(2, 2, 3)
    x = list(range(len(history['val_class_accs'][-1])))
    for i, acc in enumerate(history['val_class_accs'][-1]):
        plt.bar(i, acc, label=f'Class {i}')
    plt.title('Final Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(x)
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, save_path='confusion_matrix_optimized.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train enhanced fingering model')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for LSTM and CNN layers')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.9995, help='Beta parameter for class-balanced loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma parameter for focal loss component')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=8, help='Patience for early stopping')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for regularization')
    parser.add_argument('--clip_value', type=float, default=1.0, help='Gradient clipping value')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val = load_data(args)
    
    # Create datasets and data loaders
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    # Calculate class weights for balanced training
    class_counts = np.bincount(y_train[y_train < 10], minlength=10)
    samples_per_class = torch.tensor(class_counts, dtype=torch.float32)
    class_weights = compute_class_weights(y_train)
    
    # Create weighted sampler for handling class imbalance
    sample_weights = torch.ones(len(train_dataset), dtype=torch.float32)
    for idx, label in enumerate(y_train):
        if label < 10:  # Exclude padding class
            sample_weights[idx] = 1.0 / (class_counts[label] * 0.1)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    input_size = X_train.shape[2]
    model = EnhancedFingeringModel(
        input_size=input_size,
        seq_length=X_train.shape[1],
        hidden_size=args.hidden_size,
        dropout=0.3,
        num_classes=10
    ).to(device)
    
    # Print model summary
    print(f"Model initialized with input size: {input_size}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")
    
    # Loss function with class balancing
    criterion = ClassBalancedLoss(
        beta=args.beta,
        gamma=args.gamma,
        samples_per_class=samples_per_class,
        weight=class_weights.to(device)
    )
    
    # Optimizer with cosine annealing learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=args.learning_rate / 10
    )
    
    # Training loop
    best_val_acc = 0.0
    early_stop_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_class_accs': [],
        'learning_rates': []
    }
    
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, args.clip_value
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Evaluate on validation set
        val_loss, val_acc, class_accs, conf_matrix, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Print per-class accuracy
        for i, acc in enumerate(class_accs):
            print(f"Class {i} Acc: {acc:.4f}", end=" | ")
        print()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_class_accs'].append(class_accs)
        history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'fingering_model_optimized.pth')
            print(f"Model saved with validation accuracy: {val_acc:.4f}")
            early_stop_counter = 0
            
            # Save confusion matrix for best model
            plot_confusion_matrix(conf_matrix, save_path='confusion_matrix_optimized.png')
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{args.patience}")
        
        # Check for early stopping
        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training results
    plot_training_results(history, save_path='training_results_optimized.png')
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('fingering_model_optimized.pth'))
    
    # Final evaluation
    val_loss, val_acc, class_accs, conf_matrix, predictions, targets = evaluate(
        model, val_loader, criterion, device
    )
    
    # Print final results
    print(f"\nFinal evaluation (best model):")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print("\nPer-class accuracy:")
    for i, acc in enumerate(class_accs):
        print(f"Class {i}: {acc:.4f}")
    
    # Generate classification report
    class_report = classification_report(targets, predictions, digits=4)
    print("\nClassification Report:")
    print(class_report)
    
    # Save classification report
    with open('classification_report_optimized.txt', 'w') as f:
        f.write("Piano Fingering Model - Optimized Model Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: EnhancedFingeringModel\n")
        f.write(f"Final Validation Accuracy: {val_acc:.4f}\n\n")
        f.write("Per-Class Accuracy:\n")
        for i, acc in enumerate(class_accs):
            f.write(f"Class {i}: {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)
    
    print(f"Results saved to 'classification_report_optimized.txt'")

if __name__ == "__main__":
    main() 