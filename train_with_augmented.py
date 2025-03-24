#!/usr/bin/env python
"""
Train an optimized piano fingering model with augmented data
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Define a simpler model architecture
class EnhancedFingeringModel(nn.Module):
    def __init__(self, input_size, seq_length=10, hidden_size=128, dropout=0.3, num_classes=10):
        super(EnhancedFingeringModel, self).__init__()
        
        # Input normalization for stability
        self.input_norm = nn.LayerNorm(input_size)
        
        # Convolutional feature extraction
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # BiLSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,  # Reduced complexity
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size*2, 1)
        
        # Output classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Apply proper weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        batch_size, seq_len, features = x.size()
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Convolutional feature extraction
        x_cnn = x.transpose(1, 2)  # [batch, features, seq_len]
        x_cnn = F.relu(self.bn1(self.conv1(x_cnn)))
        x_cnn = F.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, hidden]
        
        # BiLSTM processing
        lstm_out, _ = self.lstm(x_cnn)  # [batch, seq, hidden*2]
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context)
        
        return output

class FingeringDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Import torch.nn.functional
import torch.nn.functional as F

def train_epoch(model, train_loader, optimizer, criterion, device, clip_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Store predictions and targets for analysis
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        # Display current predictions distribution to monitor class imbalance
        if batch_idx % 500 == 0:
            pred_counts = torch.bincount(predicted, minlength=10)
            pred_probs = pred_counts.float() / pred_counts.sum()
            pred_str = " ".join([f"{i}:{p:.1f}" for i, p in enumerate(pred_probs.cpu().numpy())])
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total,
                'preds': pred_str
            })
    
    return total_loss / len(train_loader), correct / total, all_predictions, all_targets

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            val_loss += criterion(output, target).item()
            
            # Get the predictions
            _, preds = torch.max(output, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Calculate per-class accuracy
    class_report = classification_report(all_targets, all_preds, output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Calculate per-class accuracy from confusion matrix
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    return val_loss / len(val_loader), accuracy, class_acc, conf_matrix, all_preds, all_targets

def plot_training_history(history, filename='training_history_augmented.png'):
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
    x = list(range(len(history['class_acc'][-1])))
    for i, acc in enumerate(history['class_acc'][-1]):
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
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(cm, filename='confusion_matrix_augmented.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train piano fingering model with augmented data')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing augmented data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=8, help='Patience for early stopping')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save model and results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}")
    
    # Load augmented data
    print(f"Loading data from {args.data_dir}...")
    try:
        X_train = np.load(os.path.join(args.data_dir, 'X_train_aug.npy'))
        y_train = np.load(os.path.join(args.data_dir, 'y_train_aug.npy'))
        X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
        
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
        # Check for dimension mismatch and fix if needed
        if X_train.shape[2] != X_val.shape[2]:
            print(f"Warning: Feature dimension mismatch - train: {X_train.shape[2]}, val: {X_val.shape[2]}")
            
            # Option 1: Use only training data
            print("Using only training data - splitting into train/val")
            # Split train data into new train/val sets
            train_size = int(0.8 * len(X_train))
            indices = np.random.permutation(len(X_train))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            
            print(f"New X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"New X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
        # Normalize the input data
        # Flatten along first dimension for normalization
        X_flat_train = X_train.reshape(-1, X_train.shape[-1])
        X_flat_val = X_val.reshape(-1, X_val.shape[-1])
        
        # Calculate mean and std
        mean = np.mean(X_flat_train, axis=0)
        std = np.std(X_flat_train, axis=0)
        std[std == 0] = 1.0  # Prevent division by zero
        
        # Normalize
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        
        # Display class distribution
        classes, counts = np.unique(y_train, return_counts=True)
        print("Class distribution:")
        for c, count in zip(classes, counts):
            print(f"  Class {c}: {count} samples ({100 * count / len(y_train):.2f}%)")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Attempting to load alternative data files...")
        try:
            # Try loading X_train.npy and X_val.npy (standard files)
            X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
            y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
            X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
            y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
            
            print(f"Successfully loaded alternative data.")
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            
            # Check for dimension mismatch and fix if needed
            if X_train.shape[2] != X_val.shape[2]:
                print(f"Warning: Feature dimension mismatch - train: {X_train.shape[2]}, val: {X_val.shape[2]}")
                
                # Option 1: Use only training data
                print("Using only training data - splitting into train/val")
                # Split train data into new train/val sets
                train_size = int(0.8 * len(X_train))
                indices = np.random.permutation(len(X_train))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                X_val = X_train[val_indices]
                y_val = y_train[val_indices]
                X_train = X_train[train_indices]
                y_train = y_train[train_indices]
                
                print(f"New X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"New X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            
            # Apply normalization
            X_flat_train = X_train.reshape(-1, X_train.shape[-1])
            X_flat_val = X_val.reshape(-1, X_val.shape[-1])
            
            # Calculate mean and std
            mean = np.mean(X_flat_train, axis=0)
            std = np.std(X_flat_train, axis=0)
            std[std == 0] = 1.0  # Prevent division by zero
            
            # Normalize
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            
            # Display class distribution
            classes, counts = np.unique(y_train, return_counts=True)
            print("Class distribution:")
            for c, count in zip(classes, counts):
                print(f"  Class {c}: {count} samples ({100 * count / len(y_train):.2f}%)")
                
        except Exception as nested_e:
            print(f"Error loading alternative data: {nested_e}")
            return
    
    # Create datasets
    train_dataset = FingeringDataset(X_train, y_train)
    val_dataset = FingeringDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compute class weights for balanced loss
    class_counts = np.bincount(y_train, minlength=10)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / sum(class_weights) * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Define loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1  # Add label smoothing
    )
    
    # Define optimizer with warm-up
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warm-up
    def lr_lambda(epoch):
        if epoch < 3:  # Warm-up for 3 epochs
            return (epoch + 1) / 3
        else:
            return 1.0  # Constant learning rate after warm-up
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'class_acc': [],
        'learning_rates': []
    }
    
    # Early stopping
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_preds, train_targets = train_epoch(
            model, train_loader, optimizer, criterion, device, args.clip_norm
        )
        
        # Evaluate
        val_loss, val_acc, class_acc, conf_matrix, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['class_acc'].append(class_acc)
        history['learning_rates'].append(current_lr)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("Class Accuracies:")
        for i, acc in enumerate(class_acc):
            print(f"  Class {i}: {acc:.4f}")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'fingering_model_augmented.pth'))
            
            # Save confusion matrix
            plot_confusion_matrix(conf_matrix, os.path.join(args.output_dir, 'confusion_matrix_augmented.png'))
            
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    # Plot training history
    plot_training_history(history, os.path.join(args.output_dir, 'training_history_augmented.png'))
    
    # Load best model and evaluate
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fingering_model_augmented.pth')))
    val_loss, val_acc, class_acc, conf_matrix, all_preds, all_targets = evaluate(
        model, val_loader, criterion, device
    )
    
    # Generate classification report
    class_report = classification_report(all_targets, all_preds, digits=4)
    print("\nClassification Report:")
    print(class_report)
    
    # Save classification report
    with open(os.path.join(args.output_dir, 'classification_report_augmented.txt'), 'w') as f:
        f.write("Piano Fingering Model - Augmented Data Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: EnhancedFingeringModel\n")
        f.write(f"Input size: {input_size}, Hidden size: {args.hidden_size}\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n\n")
        f.write("Per-class accuracy:\n")
        for i, acc in enumerate(class_acc):
            f.write(f"Class {i}: {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 