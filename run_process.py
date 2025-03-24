#!/usr/bin/env python3
"""
PIG数据集处理运行脚本
这个脚本将按顺序运行整个数据处理和模型训练流程
"""

import os
import subprocess
import time
import argparse

def run_command(command, description):
    """运行命令并打印状态"""
    print(f"\n{'=' * 80}")
    print(f"🔄 {description}")
    print(f"{'=' * 80}")
    print(f"执行命令: {command}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ 成功完成! 耗时: {elapsed_time:.2f}秒")
        return True
    else:
        print(f"❌ 命令执行失败，返回代码: {result.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行PIG钢琴指法数据处理和模型训练流程")
    parser.add_argument("--skip-prep", action="store_true", help="跳过数据集准备阶段")
    parser.add_argument("--skip-process", action="store_true", help="跳过数据处理阶段")
    parser.add_argument("--skip-train", action="store_true", help="跳过模型训练阶段")
    parser.add_argument("--dataset-path", type=str, default="PIGdata",
                       help="PIG数据集根目录路径，默认为'PIG_dataset'")
    args = parser.parse_args()
    
    print(f"\n{'*' * 80}")
    print(f"🎹 PIG钢琴指法数据处理与模型训练流程")
    print(f"{'*' * 80}")
    
    # 验证数据集路径
    dataset_dir = args.dataset_path
    fingering_dir = os.path.join(dataset_dir, "FingeringFiles")
    if not os.path.exists(fingering_dir) and not args.skip_prep:
        print(f"⚠️ 警告: 指定的数据集路径 '{fingering_dir}' 不存在!")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            print("停止执行。")
            return
    
    # 创建临时的环境变量配置文件
    with open("dataset_config.py", "w") as f:
        f.write(f"PIG_DATASET_PATH = '{dataset_dir}'\n")
        f.write(f"FINGERING_FILES_PATH = '{fingering_dir}'\n")
    
    success = True
    
    # 步骤1: 准备数据集
    if not args.skip_prep:
        prep_cmd = f"python3 dataset_prep.py --dataset-path {fingering_dir}"
        success = run_command(prep_cmd, "准备PIG数据集")
        if not success:
            print("❌ 数据集准备失败，停止执行")
            return
    else:
        print("\n⏩ 跳过数据集准备阶段")
    
    # 步骤2: 处理特征
    if not args.skip_process and success:
        success = run_command("python3 data_process.py", "处理数据特征")
        if not success:
            print("❌ 数据处理失败，停止执行")
            return
    else:
        print("\n⏩ 跳过数据处理阶段")
    
    # 步骤3: 训练模型
    if not args.skip_train and success:
        success = run_command("python3 model_training.py", "训练BiLSTM指法模型")
        if not success:
            print("❌ 模型训练失败")
            return
    else:
        print("\n⏩ 跳过模型训练阶段")
    
    # 完成
    if success:
        print(f"\n{'*' * 80}")
        print(f"🎉 整个流程已成功完成!")
        print(f"{'*' * 80}")
        print("你可以使用以下命令评估模型:")
        print("  python3 test_model.py")
        print("\n或者对新的钢琴曲进行指法预测:")
        print("  python3 predict_fingering.py path_to_midi_file")

if __name__ == "__main__":
    main() 