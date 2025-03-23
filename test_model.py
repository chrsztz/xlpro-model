import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from models import ImprovedCNNBiLSTM
from data_utils import load_pickle
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def test_model_gradient_flow(model, X_sample, y_sample, device):
    """Test gradient flow through the model"""
    model.train()
    X = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0).to(device)
    y = torch.tensor(y_sample, dtype=torch.long).unsqueeze(0).to(device)
    
    # Clear gradients
    model.zero_grad()
    
    # Forward pass
    outputs = model(X, hand_indices=2)
    loss = F.cross_entropy(outputs, y)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_info = {}
    total_norm = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            grad_info[name] = param_norm
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    
    print(f"Total gradient norm: {total_norm}")
    
    # Print top 5 largest gradients
    top_grads = sorted(grad_info.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 gradients:")
    for name, norm in top_grads:
        print(f"{name}: {norm}")
    
    # Print 5 smallest non-zero gradients
    non_zero_grads = [(n, v) for n, v in grad_info.items() if v > 0]
    bottom_grads = sorted(non_zero_grads, key=lambda x: x[1])[:5]
    print("Bottom 5 non-zero gradients:")
    for name, norm in bottom_grads:
        print(f"{name}: {norm}")
    
    return total_norm > 0

def visualize_attention(model, X_sample, device):
    """Visualize attention weights"""
    model.eval()
    X = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Forward pass with attention capture
    outputs = model(X, hand_indices=2)
    
    # Get attention weights if available in the model
    if hasattr(model, 'last_attention_weights'):
        attn_weights = model.last_attention_weights.squeeze().cpu().detach().numpy()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(attn_weights, cmap='viridis')
        plt.title('Attention Weights')
        plt.xlabel('Sequence Position')
        plt.ylabel('Attention Head')
        plt.savefig('attention_weights.png')
        plt.close()
        
        print(f"Attention visualization saved to attention_weights.png")
    else:
        print("Model does not expose attention weights for visualization")

def analyze_class_distribution(y_train, y_val, num_classes=10):
    """Analyze class distribution in training and validation sets"""
    train_dist = np.bincount(y_train[y_train != 10], minlength=num_classes)
    val_dist = np.bincount(y_val[y_val != 10], minlength=num_classes)
    
    train_percentage = train_dist / train_dist.sum() * 100
    val_percentage = val_dist / val_dist.sum() * 100
    
    print("Class Distribution (count):")
    for i in range(num_classes):
        print(f"Class {i}: Train: {train_dist[i]} ({train_percentage[i]:.2f}%), Val: {val_dist[i]} ({val_percentage[i]:.2f}%)")
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(num_classes)
    
    plt.bar(index, train_percentage, bar_width, label='Train')
    plt.bar(index + bar_width, val_percentage, bar_width, label='Validation')
    
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.title('Class Distribution')
    plt.xticks(index + bar_width/2, [str(i) for i in range(num_classes)])
    plt.legend()
    plt.savefig('class_distribution.png')
    plt.close()
    
    print(f"Class distribution visualization saved to class_distribution.png")
    
    return train_dist, val_dist

def evaluate_model(model, test_loader, device, le_fingering):
    """Evaluate model on test data and generate detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Evaluating model on test data...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch, hand_indices=2)
            
            # Ignore unlabeled samples (class 10)
            mask = y_batch != 10
            if mask.sum() > 0:
                _, preds = torch.max(outputs[mask], 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch[mask].cpu().numpy())
    
    # Classification report
    class_names = [f"{i}" for i in range(10)]
    if le_fingering:
        try:
            class_names = [f"{i}-{le_fingering.inverse_transform([i])[0]}" for i in range(10)]
        except:
            pass
    
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print(f"Confusion matrix saved to confusion_matrix.png")
    
    # Calculate and print accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    class_accuracy = []
    for i in range(10):
        mask = np.array(all_targets) == i
        if mask.sum() > 0:
            acc = np.mean(np.array(all_preds)[mask] == i)
            class_accuracy.append(acc)
            print(f"Class {class_names[i]} Accuracy: {acc:.4f}")
        else:
            class_accuracy.append(0)
            print(f"Class {class_names[i]} Accuracy: N/A (no samples)")
    
    return accuracy, class_accuracy

def check_predictions_distribution(model, loader, device):
    """Check if the model predicts only a single class"""
    model.eval()
    pred_counts = np.zeros(10)
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch, hand_indices=2)
            _, preds = torch.max(outputs, 1)
            
            for i in range(10):
                pred_counts[i] += (preds == i).sum().item()
            total += len(preds)
    
    pred_percentage = pred_counts / total * 100
    
    print("\nPrediction Distribution:")
    for i in range(10):
        print(f"Class {i}: {pred_counts[i]} samples ({pred_percentage[i]:.2f}%)")
    
    if np.max(pred_percentage) > 90:
        most_predicted = np.argmax(pred_counts)
        print(f"\nWARNING: Model is predicting mostly class {most_predicted} ({pred_percentage[most_predicted]:.2f}%)")
        print("This indicates a potential mode collapse / training issue.")
    
    return pred_counts

def main():
    # Data paths
    val_X_path = 'X_val_combined.npy'
    val_y_path = 'y_val_combined.npy'
    train_X_path = 'X_train_combined.npy'
    train_y_path = 'y_train_combined.npy'
    
    # Load data
    print("Loading data...")
    X_val = np.load(val_X_path)
    y_val = np.load(val_y_path)
    X_train = np.load(train_X_path)
    y_train = np.load(train_y_path)
    
    print(f"Data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # Analyze class distribution
    print("\nAnalyzing class distribution...")
    train_dist, val_dist = analyze_class_distribution(y_train, y_val)
    
    # Load label encoders
    print("\nLoading label encoders...")
    le_fingering = load_pickle('le_fingering.pkl')
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    input_size = X_train.shape[2]
    num_classes = 10
    
    # Initialize the improved model
    print("\nInitializing ImprovedCNNBiLSTM model...")
    model = ImprovedCNNBiLSTM(
        input_size=input_size,
        hidden_size=128,
        num_classes=num_classes,
        dropout=0.3
    )
    model.to(device)
    
    # Create datasets and loaders
    val_dataset = SimpleDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Test sample for gradient flow
    print("\nTesting gradient flow...")
    sample_idx = np.random.choice(np.where(y_train != 10)[0])
    X_sample = X_train[sample_idx]
    y_sample = y_train[sample_idx]
    grad_flow_ok = test_model_gradient_flow(model, X_sample, y_sample, device)
    
    if not grad_flow_ok:
        print("WARNING: No gradient flow detected. Check model architecture.")
    
    # Initialize optimizer for a quick training test
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training test (1 batch)
    print("\nPerforming quick training test...")
    train_dataset = SimpleDataset(X_train[:1000], y_train[:1000])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Train for just one batch to check if loss decreases
    model.train()
    batch_iter = iter(train_loader)
    X_batch, y_batch = next(batch_iter)
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    
    # First pass
    optimizer.zero_grad()
    outputs = model(X_batch, hand_indices=2)
    mask = y_batch != 10
    if mask.sum() > 0:
        loss_1 = F.cross_entropy(outputs[mask], y_batch[mask])
        loss_1.backward()
        optimizer.step()
        
        # Second pass
        optimizer.zero_grad()
        outputs = model(X_batch, hand_indices=2)
        loss_2 = F.cross_entropy(outputs[mask], y_batch[mask])
        
        print(f"Initial loss: {loss_1.item():.4f}, After one update: {loss_2.item():.4f}")
        print(f"Loss change: {loss_1.item() - loss_2.item():.4f}")
        
        if loss_2.item() < loss_1.item():
            print("✓ Training step successful: Loss decreased")
        else:
            print("⚠ Training step warning: Loss did not decrease")
    else:
        print("No labeled samples in the batch, skipping training test")
    
    # Check prediction distribution
    print("\nChecking prediction distribution before training...")
    pred_counts = check_predictions_distribution(model, val_loader, device)
    
    # Train for a few epochs to test complete training process
    print("\nRunning short training to verify complete training process...")
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, hand_indices=2)
            
            mask = y_batch != 10
            if mask.sum() > 0:
                loss = F.cross_entropy(outputs[mask], y_batch[mask])
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs[mask], 1)
                total += mask.sum().item()
                correct += (predicted == y_batch[mask]).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        
        # Evaluate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch, hand_indices=2)
                
                mask = y_batch != 10
                if mask.sum() > 0:
                    loss = F.cross_entropy(outputs[mask], y_batch[mask])
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs[mask], 1)
                    total += mask.sum().item()
                    correct += (predicted == y_batch[mask]).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Check prediction distribution after training
    print("\nChecking prediction distribution after training...")
    pred_counts = check_predictions_distribution(model, val_loader, device)
    
    # Run full evaluation with detailed metrics
    print("\nRunning full model evaluation...")
    accuracy, class_accuracy = evaluate_model(model, val_loader, device, le_fingering)
    
    # Try to save the model
    torch.save(model.state_dict(), 'improved_cnn_bilstm_test.pth')
    print("\nTest model saved to improved_cnn_bilstm_test.pth")
    
    # Print final summary
    print("\nTest Summary:")
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Recommendations
    print("\nRecommendations:")
    
    # Check if model is predicting only one class
    most_predicted = np.argmax(pred_counts)
    if pred_counts[most_predicted] / pred_counts.sum() > 0.9:
        print("- Model is primarily predicting a single class. Try the following:")
        print("  * Use a lower learning rate (0.0001)")
        print("  * Increase class weights for underrepresented classes")
        print("  * Add batch normalization for stable training")
        print("  * Initialize weights with careful scaling")
        print("  * Use gradient clipping to prevent unstable updates")
    
    # Check if accuracy is very low
    if accuracy < 0.2:
        print("- Model accuracy is very low. Try the following:")
        print("  * Verify data processing pipeline for correctness")
        print("  * Ensure input features are normalized")
        print("  * Start with a simpler model architecture")
        print("  * Analyze feature importance to identify relevant features")
    
    # Check class imbalance
    if np.max(train_dist) / np.min(train_dist[train_dist > 0]) > 10:
        print("- Severe class imbalance detected. Try the following:")
        print("  * Use stronger class weights in loss function")
        print("  * Implement data augmentation for minority classes")
        print("  * Consider stratified sampling or oversampling")
    
    print("\nNext steps:")
    print("1. Modify model_training.py to use ImprovedCNNBiLSTM")
    print("2. Train with lower learning rate and careful initialization")
    print("3. Implement proper validation during training")
    print("4. Monitor prediction distribution during training")
    
if __name__ == "__main__":
    main()