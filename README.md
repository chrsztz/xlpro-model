# XLPro Model: CNN-BiLSTM 钢琴指法生成模型

## 项目简介

基于论文《CNN-BiLSTM Hybrid Model with Physical Constraints for Automatic Piano Fingering Generation》的钢琴指法自动生成系统。

### 主要特性

- 🎹 **CNN-BiLSTM混合架构**: 结合卷积神经网络和双向长短期记忆网络
- 🦾 **物理约束集成**: 融入生物力学原理，生成更合理的指法
- 🎯 **注意力机制**: 动态聚焦关键音乐模式
- 🔄 **前向规划**: 优化全局指法连贯性
- 📊 **多维评估**: 支持Mgen、Mhigh、Msoft、Mcp等论文指标

## 项目结构

```
xlpro-model/
├── src/                          # 源代码目录
│   ├── data/                     # 数据处理模块
│   │   ├── dataset.py           # PIG数据集加载器
│   │   ├── physical_constraints.py  # 物理约束计算
│   │   └── preprocessing.py     # 数据预处理
│   ├── models/                   # 模型模块
│   │   ├── cnn_bilstm.py       # CNN-BiLSTM混合模型
│   │   ├── attention.py        # 注意力机制
│   │   ├── physical_net.py     # 物理约束网络
│   │   └── forward_planning.py # 前向规划算法
│   ├── training/                # 训练模块
│   ├── inference/               # 推理模块
│   └── utils/                   # 工具函数
├── configs/                     # 配置文件
│   └── model_config.yaml       # 模型配置
├── scripts/                     # 脚本目录
│   └── test_data_loading.py    # 数据加载测试
├── PIGdata/                     # PIG数据集
├── PianoHands.jl/              # 手部分离工具
├── requirements.txt            # Python依赖
└── README.md                   # 项目文档
```

## 安装和环境设置

### 1. 创建Python环境

```bash
# 使用conda创建虚拟环境
conda create -n xlpro-model python=3.9
conda activate xlpro-model

# 或使用venv
python -m venv xlpro-env
source xlpro-env/bin/activate  # Linux/Mac
# xlpro-env\Scripts\activate   # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python scripts/test_data_loading.py
```

## 核心技术架构

### 1. 数据表示 (Section 3.1)

按照论文规范，指法编码为:
- 右手: 1-5 (拇指到小指)
- 左手: -1到-5 (拇指到小指)  
- 无标注: 0

### 2. 特征提取 (Section 3.2)

**基础特征**:
- 归一化音高
- MIDI音符号
- 音符时长
- 黑键标识
- 和弦检测

**物理约束特征** (Section 3.3):
- 拉伸率 (Stretching Rate): `φ_stretch = |mi - mj| / |fi - fj|`
- 交叉距离 (Crossing Distance): 衡量手指交叉难度
- 手部位置 (Hand Position): 追踪手部移动
- 自然违反 (Natural Violation): 识别不自然指法
- 力度违反 (Strength Violation): 惩罚弱手指过度用力

### 3. 模型架构 (Section 3.4)

```
输入特征 → CNN特征提取 → BiLSTM序列建模 → 注意力机制 → 双分类器 → 输出
```

**CNN层**: 提取局部音乐模式
```python
H_conv = σ(BN(Conv(X_reshaped)))
```

**BiLSTM层**: 捕获长程依赖
```python
H_lstm = BiLSTM(H_conv)
```

**注意力机制**: 
```python
α = softmax(W_α tanh(W_h H_lstm))
c = Σ α_t H_lstm,t
```

**双分类器融合**:
```python
ŷ_final = (1-λ) * softmax(ŷ_main) + λ * ŷ_phys
```

### 4. 前向规划 (Section 3.5)

1. 生成初始预测: `P(f|X) = softmax(W_out · c + b_out)`
2. 选择Top-k候选: `f_top-k = argmax_k P(f|X)`
3. 物理约束评估: `s(f) = P(f|X) · (1 + PhysNet(Φ_f))`
4. 选择最优指法: `f_optimal = argmax s(f)`

## 使用方法

### 1. 数据准备

确保PIG数据集在`PIGdata/`目录下:
```
PIGdata/
├── FingeringFiles/    # 指法标注文件
├── ScorePDF/         # 乐谱PDF文件
└── List.csv          # 曲目列表
```

### 2. 训练模型

```bash
# 使用训练脚本
python scripts/train_model.py

# 或者自定义参数
python scripts/train_model.py --data_dir PIGdata --device mps
```

```python
# 或使用Python API
from src.training import create_trainer
from src.models import create_model
from src.data import PianoFingeringDataset
import yaml

# 加载配置
with open('configs/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建数据集
train_dataset = PianoFingeringDataset('PIGdata', split='train')
val_dataset = PianoFingeringDataset('PIGdata', split='val')

# 创建和训练模型
model = create_model(config['model'])
trainer = create_trainer(model, config)
history = trainer.train(train_dataset, val_dataset)
```

### 3. 推理

```bash
# 使用推理脚本
python scripts/predict_fingering.py input.mid

# 指定输出格式
python scripts/predict_fingering.py input.mid --format json --output result.json

# 生成可视化HTML
python scripts/predict_fingering.py input.mid --format html
```

```python
# 或使用Python API
from src.inference import create_predictor

# 创建预测器
predictor = create_predictor(
    model_path='experiments/cnn_bilstm_physical_constraints/best_model.pth',
    config_path='configs/model_config.yaml'
)

# 预测MIDI文件指法
result = predictor.predict_midi('input.mid')
predictor.save_result(result, 'output.mid', format='midi')

# 批量预测
output_files = predictor.batch_predict(
    ['song1.mid', 'song2.mid'], 
    'output_dir/', 
    format='json'
)
```

## 论文对应实现

| 论文章节 | 实现文件 | 说明 |
|---------|---------|------|
| 3.1 数据表示 | `src/data/dataset.py` | PIG数据格式处理 |
| 3.2 基础特征 | `src/data/feature_extractor.py` | 音乐特征提取 |
| 3.3 物理约束 | `src/data/physical_constraints.py` | 5种生物力学约束 |
| 3.4 混合网络 | `src/models/cnn_bilstm.py` | CNN-BiLSTM架构 |
| 3.5 注意力&规划 | `src/models/attention.py`, `forward_planning.py` | 注意力和前向规划 |
| 4. 实验评估 | `src/evaluation/` | 评估指标实现 |

## 评估指标

实现论文中的标准评估指标:

- **Mgen**: 通用匹配率
- **Mhigh**: 最高匹配率  
- **Msoft**: 软匹配准确率
- **Mcp**: 手位变化率

## 论文结果对比

| 模型 | Mgen | Mhigh | Msoft | Mcp |
|------|------|-------|-------|-----|
| 论文结果 | 89.2 | 83.5 | 94.1 | 0.872 |
| 本实现 | - | - | - | - |

## 手部分离集成

项目集成了`PianoHands.jl`用于MIDI文件的左右手分离:

```python
# 自动手部分离
from src.inference.hand_separation import separate_hands

left_hand, right_hand = separate_hands('input.mid')
```

## 配置说明

主要配置参数 (`configs/model_config.yaml`):

```yaml
model:
  hidden_size: 128        # 论文使用128
  num_cnn_layers: 2       # CNN层数
  num_lstm_layers: 2      # BiLSTM层数
  physical_weight: 0.3    # 物理约束权重λ
  dropout_rate: 0.3       # Dropout率

training:
  learning_rate: 0.001    # 学习率
  batch_size: 64          # 批大小
  num_epochs: 30          # 训练轮数
```

## 开发计划

- [x] 项目基础架构
- [x] 数据加载和预处理
- [x] CNN-BiLSTM模型实现
- [x] 物理约束计算
- [x] 训练pipeline
- [x] 推理系统
- [ ] 评估指标
- [ ] 可视化界面
- [ ] 性能优化

## 贡献

基于论文作者 Tianze Zhang 和 Shingyui He 的研究工作实现。

## 许可证

本项目仅用于学术研究目的。

## 联系方式

如有问题请提交Issue或联系项目维护者。 