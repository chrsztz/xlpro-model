#!/bin/bash

# 钢琴指法预测模型训练管道
# 使用方法: ./run_pipeline.sh [data_dir] [experiment_name]

set -e  # 遇到错误时退出

# 默认参数
DATA_DIR=${1:-"PIGdata"}
EXPERIMENT_NAME=${2:-"experiment_$(date +%Y%m%d_%H%M%S)"}

echo "🎹 钢琴指法预测模型训练管道"
echo "=================================="
echo "数据目录: $DATA_DIR"
echo "实验名称: $EXPERIMENT_NAME"
echo ""

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到conda，请先安装Miniconda或Anaconda"
    exit 1
fi

# 激活conda环境
echo "🔧 激活conda环境..."
if conda env list | grep -q "xlpro-model"; then
    echo "使用现有环境: xlpro-model"
    eval "$(conda shell.bash hook)"
    conda activate xlpro-model
else
    echo "创建新环境: xlpro-model"
    conda create -n xlpro-model python=3.9 -y
    eval "$(conda shell.bash hook)"
    conda activate xlpro-model
    
    # 安装基础依赖
    echo "安装PyTorch..."
    conda install pytorch torchvision torchaudio -c pytorch -y
    
    echo "安装其他依赖..."
    pip install loguru pyyaml tqdm numpy pandas matplotlib seaborn scipy scikit-learn
    pip install pretty_midi mido music21 tensorboard
fi

echo "✅ 环境准备完成"
echo ""

# 步骤1: 数据预处理
echo "📚 步骤1: 数据预处理"
echo "----------------"
python scripts/prepare_data.py \
    --data_dir "$DATA_DIR" \
    --output_dir "processed_data" \
    --report \
    --verbose

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败"
    exit 1
fi

echo "✅ 数据预处理完成"
echo ""

# 步骤2: 模型训练
echo "🚀 步骤2: 模型训练"
echo "----------------"
python scripts/train_model.py \
    --data_dir "$DATA_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs 30 \
    --batch_size 16 \
    --device auto \
    --verbose \
    --log_file "logs/${EXPERIMENT_NAME}_training.log"

if [ $? -ne 0 ]; then
    echo "❌ 模型训练失败"
    exit 1
fi

echo "✅ 模型训练完成"
echo ""

# 训练完成总结
echo "🎉 训练管道执行完成!"
echo "====================="
echo "实验名称: $EXPERIMENT_NAME"
echo "模型保存: checkpoints/$EXPERIMENT_NAME/"
echo "训练日志: logs/${EXPERIMENT_NAME}_training.log"
echo ""
echo "下一步可以使用以下命令进行推理:"
echo "python scripts/predict_fingering.py \\"
echo "  --model checkpoints/$EXPERIMENT_NAME/best_model.pth \\"
echo "  --input your_midi_file.mid \\"
echo "  --output predicted_fingering.json"