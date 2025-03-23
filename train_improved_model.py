import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
from data_utils import load_pickle  # 用于加载 LabelEncoder
from models import ImprovedCNNBiLSTM  # 使用改进的模型
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 增强的Dataset类，支持复杂的数据增强
class EnhancedFingeringDataset(Dataset):
    def __init__(self, X, y, le_hand, hand_index=2, mirror_prob=0.5, 
                 transpose_prob=0.2, transpose_range=(-2, 2), 
                 noise_prob=0.1, noise_level=0.02):
        self.X = X
        self.y = y
        self.le_hand = le_hand
        self.hand_index = hand_index
        self.mirror_prob = mirror_prob
        self.transpose_prob = transpose_prob
        self.transpose_range = transpose_range
        self.noise_prob = noise_prob
        self.noise_level = noise_level

        # 打印形状信息以便调试
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        print(f"X dtype: {self.X.dtype}, y dtype: {self.y.dtype}")

        # 计算类别分布，用于后续加权采样
        self.class_counts = np.bincount(self.y[self.y != 10], minlength=10)
        print(f"Class distribution: {self.class_counts}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample = self.X[idx].copy()
        y_sample = int(self.y[idx])  # 将y_sample转换为整数，因为它是一个单一的标签值

        # 1. 镜像增强
        if np.random.rand() < self.mirror_prob and y_sample != 10:
            # 检查最后一个时间步的手属性
            if X_sample[-1, self.hand_index] == self.le_hand.transform(['left'])[0]:
                # 将所有时间步的手属性改为右手
                X_sample[:, self.hand_index] = self.le_hand.transform(['right'])[0]
            else:
                # 将所有时间步的手属性改为左手
                X_sample[:, self.hand_index] = self.le_hand.transform(['left'])[0]

            # 指法翻转规则 - 直接应用到单一的标签值
            finger_flip = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 9, 6: 8, 7: 7, 8: 6, 9: 5}
            y_sample = finger_flip.get(y_sample, y_sample)

        # 2. 音符平移（模拟移调）- 不会影响指法
        if np.random.rand() < self.transpose_prob and y_sample != 10:
            # 假设X的第一个特征是音高特征，适当修改索引
            pitch_index = 0  # 修改为实际音高特征的索引
            transpose_amount = np.random.randint(
                self.transpose_range[0], self.transpose_range[1] + 1)
            
            # 对所有时间步的音高应用平移
            X_sample[:, pitch_index] = X_sample[:, pitch_index] + transpose_amount
            
            # 确保音高保持在合理范围内（例如MIDI音符范围21-108）
            X_sample[:, pitch_index] = np.clip(X_sample[:, pitch_index], 21, 108)

        # 3. 添加随机噪声
        if np.random.rand() < self.noise_prob and y_sample != 10:
            # 对除了分类特征（如手、指法）之外的所有特征添加噪声
            # 假设最后几列是分类特征，前面的是数值特征
            numeric_features = list(range(X_sample.shape[1]))
            numeric_features.remove(self.hand_index)  # 移除手部特征索引
            
            # 添加高斯噪声
            noise = np.random.normal(0, self.noise_level, size=(X_sample.shape[0], len(numeric_features)))
            X_sample[:, numeric_features] += noise

        # 返回转换为张量的样本
        return torch.tensor(X_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.long)

    def get_sample_weights(self):
        """获取每个样本的权重，用于加权采样"""
        weights = np.ones(len(self.y))
        for i, y_val in enumerate(self.y):
            if y_val != 10:  # 忽略未标注的样本
                # 根据类别频率的倒数计算权重
                weights[i] = 1.0 / self.class_counts[y_val]
                
        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        return weights


def train_epoch(model, loader, criterion, optimizer, device, clip_value=1.0, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    mode_collapse_detected = False
    
    # For monitoring prediction distribution during training
    batch_pred_distributions = []
    
    for i, (X_batch, y_batch) in enumerate(tqdm(loader, desc="Training")):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch, hand_indices=2)
        
        # Only compute loss on labeled samples
        mask = y_batch != 10
        if mask.sum() > 0:
            # Check for softmax outputs with very high confidence
            # This can indicate overconfidence leading to mode collapse
            with torch.no_grad():
                probs = F.softmax(outputs[mask], dim=1)
                max_probs, pred_classes = torch.max(probs, dim=1)
                
                # Monitor batch prediction distribution
                batch_preds = pred_classes.cpu().numpy()
                batch_pred_dist = np.bincount(batch_preds, minlength=10)
                batch_pred_distributions.append(batch_pred_dist)
                
                # If max probability is too high on average, add label smoothing
                high_conf = max_probs.mean() > 0.9
                
                # Check for single-class prediction in batch
                most_pred = np.argmax(batch_pred_dist)
                single_class_pred = batch_pred_dist[most_pred] / sum(batch_pred_dist) > 0.8
                
                # Update mode collapse detection
                if high_conf and single_class_pred:
                    mode_collapse_detected = True
            
            # Apply label smoothing if mode collapse is detected
            if mode_collapse_detected and i > 10:  # Skip first few batches
                # Create smooth targets
                smooth_targets = torch.zeros_like(outputs[mask])
                smooth_targets.scatter_(1, y_batch[mask].unsqueeze(1), 0.9)  # 0.9 for correct class
                # Add small probability to other classes
                smooth_targets += 0.1 / (outputs.size(1) - 1)
                # Calculate loss with smooth targets
                loss = -torch.sum(F.log_softmax(outputs[mask], dim=1) * smooth_targets) / mask.sum()
                print(f"Using label smoothing (mode collapse mitigation)")
            else:
                # Standard cross-entropy loss
                loss = criterion(outputs[mask], y_batch[mask])
            
            # Add L2 regularization on activation outputs if mode collapse detected
            if mode_collapse_detected:
                # Encourage diversity in predictions
                diversity_penalty = -0.1 * torch.mean(torch.std(F.softmax(outputs[mask], dim=1), dim=0))
                loss += diversity_penalty
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                
            optimizer.step()
            
            # Step the scheduler if it's batch-based
            if scheduler and hasattr(scheduler, "step_batch"):
                scheduler.step_batch()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs[mask], 1)
            total += mask.sum().item()
            correct += (predicted == y_batch[mask]).sum().item()
            
            # Store predictions and targets for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch[mask].cpu().numpy())
            
            # Print every 50 batches
            if (i + 1) % 50 == 0:
                print(f"Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}, Batch Acc: {(predicted == y_batch[mask]).sum().item() / mask.sum().item():.4f}")
    
    # Calculate overall metrics
    avg_loss = total_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    
    # Analyze prediction distribution to detect mode collapse
    if all_predictions:
        pred_counts = np.bincount(np.array(all_predictions), minlength=10)
        pred_percent = pred_counts / max(1, len(all_predictions)) * 100
        
        # Plot prediction distribution
        plt.figure(figsize=(10, 5))
        plt.bar(range(10), pred_percent)
        plt.xlabel('Class')
        plt.ylabel('Percentage (%)')
        plt.title('Prediction Distribution')
        plt.xticks(range(10))
        plt.savefig(f'results/pred_distribution_latest.png')
        plt.close()
        
        # If more than 90% predictions are a single class, we likely have mode collapse
        if np.max(pred_percent) > 90:
            most_pred_class = np.argmax(pred_counts)
            print(f"WARNING: Mode collapse detected - Model is predicting class {most_pred_class} for {pred_percent[most_pred_class]:.2f}% of samples")
            print(f"Prediction distribution: {pred_percent}")
            
            # Return indicator of mode collapse
            return avg_loss, accuracy, True, pred_counts
        
        # Check if prediction distribution is too unbalanced
        if np.std(pred_percent) > 30:  # High standard deviation indicates imbalance
            print(f"WARNING: Unbalanced prediction distribution detected (std={np.std(pred_percent):.2f})")
            print(f"Prediction distribution: {pred_percent}")
            
            # Return indicator of potential issues
            return avg_loss, accuracy, True, pred_counts
    
    return avg_loss, accuracy, False, None


def evaluate(model, loader, criterion, device, le_fingering=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch, hand_indices=2)
            
            # Only evaluate labeled samples
            mask = y_batch != 10
            if mask.sum() > 0:
                loss = criterion(outputs[mask], y_batch[mask])
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs[mask], 1)
                total += mask.sum().item()
                correct += (predicted == y_batch[mask]).sum().item()
                
                # Collect predictions and targets
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch[mask].cpu().numpy())
    
    # Calculate overall metrics
    avg_loss = total_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    
    # If we have predictions, compute detailed metrics
    if all_predictions:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Per-class accuracy
        class_accuracies = []
        for cls in range(10):
            mask = all_targets == cls
            if mask.sum() > 0:
                cls_acc = np.mean(all_predictions[mask] == cls)
                class_accuracies.append(cls_acc)
            else:
                class_accuracies.append(0.0)
        
        # Generate confusion matrix once per epoch
        if le_fingering:
            try:
                class_names = [f"{i}-{le_fingering.inverse_transform([i])[0]}" for i in range(10)]
            except:
                class_names = [str(i) for i in range(10)]
        else:
            class_names = [str(i) for i in range(10)]
        
        cm = confusion_matrix(all_targets, all_predictions, labels=range(10))
        
        # Print detailed class metrics
        print("\nPer-class Accuracy:")
        for i, acc in enumerate(class_accuracies):
            print(f"Class {class_names[i]}: {acc:.4f}")
        
        return avg_loss, accuracy, class_accuracies, cm
    else:
        return avg_loss, accuracy, [0.0] * 10, None


def save_training_plots(history, output_dir='results'):
    """Save training history plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_history.png')
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{output_dir}/accuracy_history.png')
    plt.close()
    
    # Plot per-class accuracies for last epoch
    plt.figure(figsize=(12, 6))
    class_accs = history['val_class_accs'][-1] if history['val_class_accs'] else [0] * 10
    x = np.arange(10)
    plt.bar(x, class_accs)
    plt.xticks(x, [str(i) for i in range(10)])
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-class Accuracy (Last Epoch)')
    plt.savefig(f'{output_dir}/class_accuracy.png')
    plt.close()
    
    # Print summary of best results
    best_epoch = np.argmax(history['val_acc'])
    print(f"\nBest model at epoch {best_epoch+1}:")
    print(f"  Validation Accuracy: {history['val_acc'][best_epoch]:.4f}")
    print(f"  Validation Loss: {history['val_loss'][best_epoch]:.4f}")


def plot_confusion_matrix(cm, class_names, output_dir='results'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()


def main():
    # Data paths
    train_X_path = 'X_train_combined.npy'
    train_y_path = 'y_train_combined.npy'
    val_X_path = 'X_val_combined.npy'
    val_y_path = 'y_val_combined.npy'
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Print data info
    print(f"Loading training and validation data...")
    
    # Load data
    X_train = np.load(train_X_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_X_path)
    y_val = np.load(val_y_path)
    
    print(f"Data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # Load LabelEncoders
    le_hand = load_pickle('le_hand.pkl')
    le_fingering = load_pickle('le_fingering.pkl')
    
    # Separate labeled and unlabeled data
    labeled_idx = y_train != 10
    X_train_labeled = X_train[labeled_idx]
    y_train_labeled = y_train[labeled_idx]
    X_train_unlabeled = X_train[~labeled_idx]
    
    print(f"Labeled data: {len(X_train_labeled)}, Unlabeled data: {len(X_train_unlabeled)}")
    
    # Model parameters
    input_size = X_train.shape[2]
    hidden_size = 128  # Smaller hidden size for stability
    num_classes = 10
    dropout = 0.3
    
    print(f"Model parameters - input_size: {input_size}, hidden_size: {hidden_size}, num_classes: {num_classes}")
    
    # Initialize the ImprovedCNNBiLSTM model with careful initialization
    model = ImprovedCNNBiLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Apply careful weight initialization to prevent mode collapse
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Use Xavier/Glorot initialization for linear layers
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # Lower gain for more stable gradients
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.001)  # Small positive bias to prevent dead neurons
        elif isinstance(m, nn.Conv1d):
            # Use Kaiming initialization for conv layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.001)
        elif isinstance(m, nn.LSTM):
            # Careful LSTM initialization
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param, gain=0.5)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param, gain=0.5)  # Orthogonal for recurrent weights
                elif 'bias' in name:
                    param.data.fill_(0.0)
                    # Set forget gate bias to 1.0 (helps learning long-term dependencies)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    # Apply the initialization
    model.apply(init_weights)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} total parameters")
    print(f"Model has {trainable_params:,} trainable parameters")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Create enhanced dataset
    train_dataset = EnhancedFingeringDataset(
        X_train_labeled, 
        y_train_labeled, 
        le_hand, 
        hand_index=2,
        mirror_prob=0.5,
        transpose_prob=0.2,
        transpose_range=(-2, 2),
        noise_prob=0.1
    )
    
    val_dataset = EnhancedFingeringDataset(
        X_val, y_val, le_hand, 
        hand_index=2, 
        mirror_prob=0.0,  # No augmentation for validation
        transpose_prob=0.0,
        noise_prob=0.0
    )
    
    # Calculate class weights for loss function with stronger weighting for minority classes
    class_counts = np.bincount(y_train_labeled, minlength=10)
    print(f"Raw class counts: {class_counts}")
    
    # More aggressive weighting to combat class imbalance
    # Using square root of reciprocal instead of direct reciprocal
    # This gives more weight to minority classes without making weights too extreme
    class_weights = np.sqrt(np.max(class_counts) / (class_counts + 1))
    
    # Smooth weights to avoid extreme values but maintain strong signal
    # Min weight = 0.7, Max weight = 1.5 (empirically determined)
    min_weight, max_weight = 0.7, 1.5
    class_weights = min_weight + (max_weight - min_weight) * (class_weights - np.min(class_weights)) / (np.max(class_weights) - np.min(class_weights))
    
    # Extra boost to least represented classes
    for i in range(len(class_weights)):
        if class_counts[i] / class_counts.sum() < 0.05:  # Less than 5% of data
            class_weights[i] *= 1.2  # 20% boost
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"Final class weights: {class_weights}")
    
    # Save class weights for reference
    np.save('results/class_weights.npy', class_weights)
    
    # Create weighted sampler for class balance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(train_dataset), 
        replacement=True
    )
    
    # Data loaders
    batch_size = 32  # Smaller batch size for stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function with class weights
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=10)
    
    # Optimizer with very small learning rate and weight decay
    learning_rate = 0.00005  # Very small learning rate to prevent mode collapse
    weight_decay = 0.005     # Moderate regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),   # Modified beta2 for better convergence
        eps=1e-8             # For numerical stability
    )
    
    # LR scheduler with ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',          # Monitor accuracy (higher is better)
        factor=0.5,          # Reduce LR by half when plateauing
        patience=3,          # Wait 3 epochs before reducing LR
        verbose=True,
        min_lr=1e-6          # Don't reduce LR below this value
    )
    
    # Training parameters
    num_epochs = 30
    best_val_acc = 0.0
    patience = 8             # Early stopping patience
    counter = 0              # Early stopping counter
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_class_accs': [],
        'learning_rates': []
    }
    
    # Create model checkpoint directory
    os.makedirs('models', exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    # Track mode collapse instances
    mode_collapse_count = 0
    mode_collapse_threshold = 3  # Number of allowed mode collapse detections before intervention
    
    for epoch in range(num_epochs):
        # Save current learning rate
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Train
        train_loss, train_acc, mode_collapse, pred_counts = train_epoch(
            model, train_loader, criterion, optimizer, device, clip_value=1.0
        )
        
        # Update mode collapse tracking
        if mode_collapse:
            mode_collapse_count += 1
            print(f"Mode collapse detected in epoch {epoch+1} (count: {mode_collapse_count}/{mode_collapse_threshold})")
            
            # If mode collapse is persistent, take corrective action
            if mode_collapse_count >= mode_collapse_threshold:
                print("Persistent mode collapse detected - applying corrective measures:")
                
                # 1. Reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  - Reduced learning rate to {optimizer.param_groups[0]['lr']}")
                
                # 2. Reinitialize last layer with better initialization
                print("  - Reinitializing classification layer")
                nn.init.xavier_uniform_(model.fc2.weight, gain=0.1)  # Very small gain
                nn.init.zeros_(model.fc2.bias)  # Start with zero bias
                
                # 3. Analyze which class is dominating and adjust class weights
                if pred_counts is not None:
                    dominant_class = np.argmax(pred_counts)
                    print(f"  - Dominant class: {dominant_class}")
                    
                    # Reduce weight for dominant class, increase for others
                    new_class_weights = class_weights.copy()
                    new_class_weights[dominant_class] *= 0.5  # Halve the weight of dominant class
                    # Increase weights of non-dominant classes
                    for i in range(10):
                        if i != dominant_class:
                            new_class_weights[i] *= 1.2  # Increase by 20%
                    
                    # Normalize again
                    new_class_weights = new_class_weights / new_class_weights.sum() * len(new_class_weights)
                    print(f"  - Updated class weights: {new_class_weights}")
                    
                    # Update criterion with new weights
                    class_weights_tensor = torch.tensor(new_class_weights, dtype=torch.float32).to(device)
                    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=10)
                
                # Reset counter after intervention
                mode_collapse_count = 0
                
                # Save intervention checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': None,  # Not computed yet
                    'val_acc': None,   # Not computed yet
                    'history': history,
                    'intervention': True
                }, f'models/intervention_epoch_{epoch+1}.pth')
        
        # Evaluate
        val_loss, val_acc, class_accs, confusion = evaluate(model, val_loader, criterion, device, le_fingering)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_class_accs'].append(class_accs)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Print per-class evaluation summary
        print("Per-class accuracy summary:")
        for i, acc in enumerate(class_accs[:5]):  # First 5 classes
            print(f"  Class {i}: {acc:.4f}", end="  ")
        print()
        for i, acc in enumerate(class_accs[5:]):  # Last 5 classes
            print(f"  Class {i+5}: {acc:.4f}", end="  ")
        print()
        
        # Step LR scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        # Save confusion matrix for this epoch
        if confusion is not None:
            class_names = [str(i) for i in range(10)]
            if le_fingering:
                try:
                    class_names = [f"{i}-{le_fingering.inverse_transform([i])[0]}" for i in range(10)]
                except:
                    pass
            
            # Create epoch-specific directory
            os.makedirs(f'results/epoch_{epoch+1}', exist_ok=True)
            plot_confusion_matrix(confusion, class_names, output_dir=f'results/epoch_{epoch+1}')
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"New best validation accuracy: {best_val_acc:.4f}, model saved")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
            
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint at the end of each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': history
        }, f'models/checkpoint_epoch_{epoch+1}.pth')
        
        # Save training plots after each epoch
        save_training_plots(history)
    
    # Save final model
    torch.save(model.state_dict(), 'models/final_model.pth')
    print("Training completed, final model saved")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('models/best_model.pth'))
    val_loss, val_acc, class_accs, confusion = evaluate(model, val_loader, criterion, device, le_fingering)
    print(f"\nFinal evaluation (best model) - Validation Accuracy: {val_acc:.4f}")
    
    # Save model as fingering_model_final.pth for compatibility with existing code
    torch.save(model.state_dict(), 'fingering_model_final.pth')
    print("Best model saved as fingering_model_final.pth")
    
    # Generate final report
    with open('results/training_report.txt', 'w') as f:
        f.write("Piano Fingering Model Training Report\n")
        f.write("==================================\n\n")
        f.write(f"Model: ImprovedCNNBiLSTM\n")
        f.write(f"Input size: {input_size}\n")
        f.write(f"Hidden size: {hidden_size}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write(f"Parameters: {trainable_params:,}\n\n")
        f.write(f"Training data: {len(X_train_labeled)} samples\n")
        f.write(f"Validation data: {len(X_val)} samples\n\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Final validation accuracy: {val_acc:.4f}\n\n")
        f.write("Per-class accuracy:\n")
        for i, acc in enumerate(class_accs):
            class_name = str(i)
            if le_fingering:
                try:
                    class_name = f"{i}-{le_fingering.inverse_transform([i])[0]}"
                except:
                    pass
            f.write(f"  Class {class_name}: {acc:.4f}\n")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()