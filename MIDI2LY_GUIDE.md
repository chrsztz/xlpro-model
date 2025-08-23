# MIDI2LY 指法预测链条使用指南

## 🎯 方案优势

使用 `midi2ly` 工具相比手动生成LilyPond的优势：

### ✅ midi2ly工具的优势
- **音乐理论正确**: LilyPond官方工具，自动处理调号、拍号、时值量化
- **格式完整**: 支持复杂的音乐结构和记号
- **稳定可靠**: 经过大量测试，处理各种MIDI格式
- **维护成本低**: 不需要我们维护音乐转换逻辑

### ✅ 我们的增强
- **智能指法添加**: 解析LilyPond代码，精确添加指法信息
- **左右手识别**: 自动识别声部并分配相应指法
- **错误处理**: 完善的异常处理和回退机制

## 🛠️ 技术实现

### 完整流程
```
MIDI文件 → CNN-BiLSTM预测 → 指法结果 → midi2ly转换 → 解析LilyPond → 添加指法 → PDF编译
```

### 核心组件

1. **`src/inference/midi2ly_processor.py`** - 核心处理器
2. **`scripts/predict_with_midi2ly.py`** - 完整预测脚本  
3. **`scripts/test_midi2ly.py`** - 测试和验证脚本

## 🚀 使用方法

### 1. 环境准备
```bash
# 安装LilyPond (包含midi2ly工具)
brew install lilypond  # macOS
# 或
sudo apt-get install lilypond  # Ubuntu

# 验证安装
midi2ly --version
lilypond --version
```

### 2. 测试midi2ly功能
```bash
# 激活conda环境
conda activate xlpro-model

# 测试midi2ly基本功能
python scripts/test_midi2ly.py
```

### 3. 只测试工具链 (不需要训练模型)
```bash
# 测试midi2ly处理器
python scripts/predict_with_midi2ly.py --test-only
```

### 4. 完整预测 (需要训练好的模型)
```bash
# 使用训练好的模型进行完整预测
python scripts/predict_with_midi2ly.py input.mid \
    --model experiments/your_model/best_model.pth \
    --output lilypond_results
```

## 📁 输出文件

运行完整预测后，在输出目录中会生成：

```
lilypond_results/
├── input_with_fingerings.ly    # 带指法的LilyPond源码
├── input_with_fingerings.pdf   # 最终的PDF乐谱
└── input_fingering_data.json   # 指法数据(备用)
```

## 🎼 LilyPond指法效果

### 输入MIDI
```
C4 D4 E4 F4 G4 (右手旋律)
C3 G3 (左手伴奏)
```

### midi2ly输出 (基础)
```lilypond
\new Staff {
  \clef treble
  c'4 d'4 e'4 f'4 g'4
}
\new Staff {
  \clef bass
  c4 g4
}
```

### 添加指法后
```lilypond
\new Staff {
  \clef treble
  c'4-1 d'4-2 e'4-3 f'4-4 g'4-5
}
\new Staff {
  \clef bass
  c4-5 g4-1
}
```

### 最终PDF效果
- 📄 专业级乐谱排版
- 🖐️ 清晰的指法数字标记
- 🎹 正确的调号、拍号、时值
- 📏 自动分页和美观排版

## 🔧 技术细节

### midi2ly参数优化
```bash
midi2ly \
  --duration-quant 8 \    # 八分音符量化
  --allow-tuplet \        # 允许连音符
  --output score.ly \     # 输出文件
  input.mid              # 输入MIDI
```

### 指法添加算法
```python
# 1. 解析midi2ly生成的音符
note_pattern = r'([a-g](?:is|es)?[,\']*\d+)'

# 2. 按时间顺序匹配指法
for note_match in note_matches:
    note_str = note_match.group(1)
    finger = get_fingering_for_note(note_str, time, hand)
    fingered_note = f"{note_str}-{finger}"

# 3. 替换原始音符
modified_ly_code = re.sub(pattern, replacement, original_code)
```

### 错误处理策略
```python
try:
    # 尝试使用midi2ly
    ly_path = convert_with_midi2ly(midi_file)
except Exception:
    # 回退到手动生成
    ly_path = generate_manually(midi_file)
```

## 🎯 质量保证

### 音乐理论正确性
- ✅ **调号识别**: midi2ly自动分析调性
- ✅ **拍号检测**: 自动识别时间签名
- ✅ **时值量化**: 智能量化到合适的音符时值
- ✅ **声部分离**: 正确处理左右手声部

### 指法准确性
- ✅ **时间匹配**: 基于onset_time精确匹配
- ✅ **音高对应**: 确保指法与正确音符关联
- ✅ **手部识别**: 自动识别左右手声部
- ✅ **边界处理**: 处理无指法标注的情况

## 🚀 实际使用示例

### 完整工作流程
```bash
# 1. 训练模型 (如果还没有)
python scripts/train_model.py --epochs 10 --experiment_name quick_test

# 2. 预测并生成乐谱
python scripts/predict_with_midi2ly.py your_song.mid \
    --model experiments/quick_test/best_model.pth \
    --output results

# 3. 查看结果
open results/your_song_with_fingerings.pdf
```

### 批量处理
```bash
# 处理多个MIDI文件
for midi_file in *.mid; do
    python scripts/predict_with_midi2ly.py "$midi_file" \
        --model experiments/your_model/best_model.pth \
        --output "results/$(basename "$midi_file" .mid)"
done
```

## 🔍 故障排除

### 常见问题

1. **midi2ly未找到**
   ```bash
   which midi2ly
   # 如果没有输出，重新安装LilyPond
   brew reinstall lilypond
   ```

2. **转换失败**
   - 检查MIDI文件是否有效
   - 尝试简化的MIDI文件测试
   - 查看midi2ly错误信息

3. **指法显示异常**
   - 检查指法数值范围
   - 验证音符匹配逻辑
   - 查看LilyPond源码

4. **PDF编译失败**
   - 检查LilyPond语法
   - 确保有足够磁盘空间
   - 查看编译错误信息

### 调试步骤
```bash
# 1. 测试基本功能
python scripts/test_midi2ly.py

# 2. 测试处理器
python scripts/predict_with_midi2ly.py --test-only

# 3. 使用简单MIDI测试
python scripts/predict_with_midi2ly.py simple_test.mid

# 4. 手动检查LilyPond代码
cat output/score_with_fingerings.ly
```

## 📊 性能对比

| 方案 | 音乐理论 | 开发难度 | 维护成本 | 输出质量 | 推荐度 |
|------|----------|----------|----------|----------|--------|
| 手动生成 | 需要实现 | 高 | 高 | 中等 | ⭐⭐ |
| **midi2ly + 后处理** | **自动正确** | **中等** | **低** | **优秀** | **⭐⭐⭐⭐⭐** |

## 🎉 预期效果

使用这个方案，您将得到：
- 🎵 **正确的音乐记号**: 调号、拍号、时值都由midi2ly自动处理
- 🖐️ **精确的指法标记**: AI预测的指法准确添加到对应音符
- 📄 **专业级PDF**: LilyPond生成的高质量乐谱
- 🔧 **易于维护**: 基于成熟工具，减少bug和维护工作

这确实是比手动生成更优的解决方案！


