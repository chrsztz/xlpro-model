"""
快速测试脚本

验证环境和基础功能是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """测试必要的包导入"""
    print("🔍 测试包导入...")
    
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
        
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
        
        import yaml
        print("  ✅ PyYAML")
        
        from loguru import logger
        print("  ✅ Loguru")
        
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False

def test_project_structure():
    """测试项目结构"""
    print("🏗️ 测试项目结构...")
    
    required_dirs = [
        "src",
        "src/data",
        "src/models", 
        "src/training",
        "src/inference",
        "configs"
    ]
    
    missing = []
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            missing.append(dir_name)
        else:
            print(f"  ✅ {dir_name}/")
    
    if missing:
        print(f"  ❌ 缺少目录: {missing}")
        return False
    
    return True

def test_config():
    """测试配置文件"""
    print("⚙️ 测试配置文件...")
    
    try:
        import yaml
        config_path = project_root / "configs" / "model_config.yaml"
        
        if not config_path.exists():
            print(f"  ❌ 配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['data', 'model', 'training', 'experiment']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"  ❌ 配置缺少键: {missing_keys}")
            return False
        
        print("  ✅ 配置文件格式正确")
        return True
        
    except Exception as e:
        print(f"  ❌ 配置文件错误: {e}")
        return False

def test_data_loading():
    """测试数据加载(如果数据存在)"""
    print("📚 测试数据加载...")
    
    try:
        from src.data.dataset import PianoFingeringDataset
        
        # 检查数据目录
        data_dir = project_root / "PIGdata"
        if not data_dir.exists():
            print("  🟡 数据目录不存在，跳过数据加载测试")
            return True
        
        # 尝试创建小数据集
        dataset = PianoFingeringDataset(
            data_dir=data_dir,
            split="train",
            sequence_length=10,
            max_pieces=1  # 只加载1个文件
        )
        
        if len(dataset) > 0:
            print(f"  ✅ 数据加载成功，样本数: {len(dataset)}")
            return True
        else:
            print("  🟡 数据集为空")
            return True
            
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("🧠 测试模型创建...")
    
    try:
        from src.models.cnn_bilstm import create_model
        
        # 简单的模型配置
        model_config = {
            'input_dim': 5,
            'hidden_size': 32,  # 小一点用于测试
            'num_cnn_layers': 1,
            'num_lstm_layers': 1,
            'num_classes': 11,
            'dropout_rate': 0.1,
            'physical_features_dim': 5,
            'physical_weight': 0.3,
            'use_attention': True,
            'use_batch_norm': True
        }
        
        model = create_model(model_config)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✅ 模型创建成功，参数数量: {total_params:,}")
        return True
        
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        return False

def main():
    print("🎹 钢琴指法预测项目 - 快速测试")
    print("=" * 50)
    
    tests = [
        ("包导入", test_imports),
        ("项目结构", test_project_structure),
        ("配置文件", test_config),
        ("数据加载", test_data_loading),
        ("模型创建", test_model_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print()
        if test_func():
            passed += 1
        else:
            print(f"  ⚠️ {test_name} 测试未通过")
    
    print()
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境配置正确。")
        print("现在可以运行完整的训练管道:")
        print("  ./run_pipeline.sh PIGdata my_experiment")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查环境配置。")
        return 1

if __name__ == "__main__":
    exit(main())



