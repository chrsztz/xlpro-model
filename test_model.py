import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from models import EnhancedBiLSTMWithAttention
from data_utils import load_pickle
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def plot_confusion_matrix(cm, class_names, output_dir='results'):
    """绘制混淆矩阵"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

def main():
    # 加载数据和模型
    print("加载验证数据...")
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    
    print(f"验证数据形状: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # 加载标签编码器
    le_fingering = load_pickle('le_fingering.pkl')
    class_names = [str(i) for i in range(10)]  # 指法类别名称
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model_path = 'fingering_bilstm_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    print(f"加载模型: {model_path}")
    
    # 初始化模型
    input_size = X_val.shape[2]
    hidden_size = 256
    num_layers = 3
    num_classes = 10
    dropout = 0.3
    
    model = EnhancedBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 创建数据加载器
    val_dataset = SimpleDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 评估模型
    print("评估模型...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 计算每个类别的准确率
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    
    # 打印详细的分类报告
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # 打印总体准确率
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\n总体准确率: {accuracy:.4f}")
    
    # 打印每个类别的准确率
    print("\n每个类别的准确率:")
    for i, acc in enumerate(class_accuracy):
        print(f"指法 {i}: {acc:.4f}")
    
    # 可视化混淆矩阵
    plot_confusion_matrix(cm, class_names)
    print("混淆矩阵已保存到 results/confusion_matrix.png")

if __name__ == "__main__":
    main()