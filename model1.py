from music21 import converter, note, articulations

# 加载mxl文件
score = converter.parse('example.mxl')

# 遍历所有音符
for n in score.flat.notes:
    # 检查音符是否有articulations
    if n.articulations:
        for articulation in n.articulations:
            # 查找fingering标记
            if isinstance(articulation, articulations.Fingering):
                # 输出指法信息
                print(f"音符: {n}, 指法: {articulation.fingerNumber}")
