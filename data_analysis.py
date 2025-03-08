import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import os
from data_utils import load_pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_data():
    """加载训练和验证数据"""
    print("加载数据...")

    # 检查文件是否存在
    train_X_path = 'X_train_combined.npy'
    train_y_path = 'y_train_combined.npy'
    val_X_path = 'X_val_combined.npy'
    val_y_path = 'y_val_combined.npy'

    for path in [train_X_path, train_y_path, val_X_path, val_y_path]:
        if not os.path.exists(path):
            print(f"警告: 文件 {path} 不存在")

    try:
        X_train = np.load(train_X_path, mmap_mode='r')
        y_train = np.load(train_y_path, mmap_mode='r')
        X_val = np.load(val_X_path, mmap_mode='r')
        y_val = np.load(val_y_path, mmap_mode='r')

        # 加载LabelEncoder
        try:
            le_fingering = load_pickle('le_fingering.pkl')
            print("加载了指法编码器")
        except:
            print("找不到指法编码器文件")
            le_fingering = None

        return X_train, y_train, X_val, y_val, le_fingering
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None, None, None, None


def analyze_labels(y_train, y_val, le_fingering=None):
    """分析标签分布"""
    print("\n===== 标签分布分析 =====")

    # 训练集标签分布
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    train_distribution = dict(zip(train_labels, train_counts))
    print(f"训练集标签分布: {train_distribution}")

    # 如果有Encoder，转换标签名称
    if le_fingering:
        label_names = {}
        for label in train_labels:
            if hasattr(le_fingering, 'inverse_transform'):
                original_label = le_fingering.inverse_transform([label])[0]
                label_names[label] = f"{label}(指法:{original_label})"
            else:
                label_names[label] = str(label)
    else:
        label_names = {label: str(label) for label in train_labels}

    # 验证集标签分布
    val_labels, val_counts = np.unique(y_val, return_counts=True)
    val_distribution = dict(zip(val_labels, val_counts))
    print(f"验证集标签分布: {val_distribution}")

    # 计算标注比例
    labeled_train = np.sum(y_train != 10)
    unlabeled_train = np.sum(y_train == 10)
    print(f"训练集: 已标注样本 {labeled_train} ({labeled_train / len(y_train) * 100:.2f}%), "
          f"未标注样本 {unlabeled_train} ({unlabeled_train / len(y_train) * 100:.2f}%)")

    labeled_val = np.sum(y_val != 10)
    unlabeled_val = np.sum(y_val == 10)
    print(f"验证集: 已标注样本 {labeled_val} ({labeled_val / len(y_val) * 100:.2f}%), "
          f"未标注样本 {unlabeled_val} ({unlabeled_val / len(y_val) * 100:.2f}%)")

    # 可视化训练集标签分布
    plt.figure(figsize=(12, 6))

    # 只统计有效标签（非10）
    valid_labels = [l for l in train_labels if l != 10]
    valid_counts = [train_distribution[l] for l in valid_labels]

    plt.bar([label_names.get(l, str(l)) for l in valid_labels], valid_counts)
    plt.title('训练集有效标签分布')
    plt.xlabel('标签')
    plt.ylabel('样本数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    print(f"标签分布图已保存为 'label_distribution.png'")


def analyze_features(X_train, y_train):
    """分析特征分布"""
    print("\n===== 特征分布分析 =====")

    # 基本统计信息
    print(f"特征形状: {X_train.shape}")
    print(f"序列长度: {X_train.shape[1]}")
    print(f"特征维度: {X_train.shape[2]}")

    # 计算每个特征的基本统计量
    print("计算特征统计量...")

    # 获取已标注和未标注样本
    labeled_mask = y_train != 10
    X_labeled = X_train[labeled_mask]
    X_unlabeled = X_train[~labeled_mask]

    print(f"已标注样本: {X_labeled.shape[0]}")
    print(f"未标注样本: {X_unlabeled.shape[0]}")

    # 对特征进行统计
    feature_stats = []

    # 只分析前10个和后10个特征维度，避免分析太多
    feature_indices = list(range(0, 10)) + list(range(X_train.shape[2] - 10, X_train.shape[2]))

    for i in feature_indices:
        # 取最后一个时间步的特征
        labeled_values = X_labeled[:, -1, i].flatten()
        unlabeled_values = X_unlabeled[:, -1, i].flatten() if len(X_unlabeled) > 0 else np.array([])

        labeled_mean = np.mean(labeled_values)
        labeled_std = np.std(labeled_values)
        labeled_min = np.min(labeled_values)
        labeled_max = np.max(labeled_values)

        unlabeled_mean = np.mean(unlabeled_values) if len(unlabeled_values) > 0 else np.nan
        unlabeled_std = np.std(unlabeled_values) if len(unlabeled_values) > 0 else np.nan

        # 检查是否有异常值
        has_nan = np.isnan(labeled_values).any() or (len(unlabeled_values) > 0 and np.isnan(unlabeled_values).any())
        has_inf = np.isinf(labeled_values).any() or (len(unlabeled_values) > 0 and np.isinf(unlabeled_values).any())

        # 计算分布差异
        if len(unlabeled_values) > 0:
            try:
                ks_statistic, ks_pvalue = stats.ks_2samp(labeled_values, unlabeled_values)
            except:
                ks_statistic, ks_pvalue = np.nan, np.nan
        else:
            ks_statistic, ks_pvalue = np.nan, np.nan

        feature_stats.append({
            'feature_idx': i,
            'labeled_mean': labeled_mean,
            'labeled_std': labeled_std,
            'labeled_min': labeled_min,
            'labeled_max': labeled_max,
            'unlabeled_mean': unlabeled_mean,
            'unlabeled_std': unlabeled_std,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue
        })

    # 转换为DataFrame以便查看
    stats_df = pd.DataFrame(feature_stats)
    print("\n特征统计信息摘要:")
    print(stats_df[['feature_idx', 'labeled_mean', 'labeled_std', 'has_nan', 'has_inf']].head(10))

    # 保存完整统计信息
    stats_df.to_csv('feature_statistics.csv', index=False)
    print("完整特征统计已保存到 'feature_statistics.csv'")

    # 检查异常值
    problem_features = stats_df[(stats_df['has_nan'] == True) | (stats_df['has_inf'] == True)]
    if len(problem_features) > 0:
        print(f"\n警告: 发现{len(problem_features)}个包含NaN或Inf的特征!")
        print(problem_features[['feature_idx', 'has_nan', 'has_inf']])

    # 检查标注/未标注数据分布差异较大的特征
    if not np.isnan(stats_df['ks_pvalue']).all():
        significant_diff = stats_df[stats_df['ks_pvalue'] < 0.001]
        if len(significant_diff) > 0:
            print(f"\n标注与未标注数据分布差异显著的特征数量: {len(significant_diff)}")
            print(significant_diff[['feature_idx', 'ks_statistic', 'ks_pvalue']].head(10))

    # 可视化部分特征分布
    plt.figure(figsize=(15, 10))

    # 选取一些有代表性的特征进行可视化
    sample_features = [0, 1, 2, 5] if X_train.shape[2] > 5 else list(range(X_train.shape[2]))

    for i, feature_idx in enumerate(sample_features):
        plt.subplot(2, 2, i + 1)

        # 获取特征值
        labeled_values = X_labeled[:, -1, feature_idx].flatten()
        unlabeled_values = X_unlabeled[:, -1, feature_idx].flatten() if len(X_unlabeled) > 0 else np.array([])

        # 绘制直方图
        plt.hist(labeled_values, bins=50, alpha=0.5, label='已标注')
        if len(unlabeled_values) > 0:
            plt.hist(unlabeled_values, bins=50, alpha=0.5, label='未标注')

        plt.title(f'特征 {feature_idx} 分布')
        plt.xlabel('值')
        plt.ylabel('频率')
        plt.legend()

    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    print("特征分布图已保存为 'feature_distributions.png'")


def analyze_feature_correlations(X_train, y_train):
    """分析特征与标签的相关性"""
    print("\n===== 特征相关性分析 =====")

    # 只使用已标注数据
    labeled_mask = y_train != 10
    X_labeled = X_train[labeled_mask]
    y_labeled = y_train[labeled_mask]

    # 将3D特征张量转为2D (样本数 x 特征数)
    # 只使用最后一个时间步的特征
    X_flat = X_labeled[:, -1, :]

    print(f"计算特征与标签的相关性 (形状: {X_flat.shape})...")

    # 特征太多可能导致计算缓慢，取前20个特征
    n_features = min(20, X_flat.shape[1])
    X_subset = X_flat[:, :n_features]

    # 计算特征之间的相关性
    try:
        corr_matrix = np.corrcoef(X_subset, rowvar=False)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm',
                    xticklabels=range(n_features),
                    yticklabels=range(n_features))
        plt.title('特征相关性热图 (前20个特征)')
        plt.tight_layout()
        plt.savefig('feature_correlation.png')
        print("特征相关性热图已保存为 'feature_correlation.png'")
    except Exception as e:
        print(f"计算特征相关性时出错: {e}")

    # 尝试计算特征与标签的相关性
    try:
        # 将特征和标签合并为一个数组
        combined = np.column_stack((X_subset, y_labeled))
        corr_with_target = np.corrcoef(combined, rowvar=False)[-1, :-1]

        # 按相关性绝对值排序
        sorted_indices = np.argsort(np.abs(corr_with_target))[::-1]

        print("\n特征与标签的相关性 (前10个最相关特征):")
        for i, idx in enumerate(sorted_indices[:10]):
            print(f"特征 {idx}: 相关性 = {corr_with_target[idx]:.4f}")

        # 可视化特征与标签的相关性
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(corr_with_target)), np.abs(corr_with_target)[sorted_indices])
        plt.xticks(range(len(corr_with_target)), sorted_indices, rotation=90)
        plt.title('特征与标签的相关性')
        plt.xlabel('特征索引')
        plt.ylabel('相关性绝对值')
        plt.tight_layout()
        plt.savefig('feature_label_correlation.png')
        print("特征与标签相关性图已保存为 'feature_label_correlation.png'")
    except Exception as e:
        print(f"计算特征与标签相关性时出错: {e}")


def visualize_high_dim_features(X_train, y_train):
    """使用降维方法可视化高维特征"""
    print("\n===== 高维特征可视化 =====")

    # 只使用已标注数据
    labeled_mask = y_train != 10
    X_labeled = X_train[labeled_mask]
    y_labeled = y_train[labeled_mask]

    # 取最后一个时间步的特征
    X_flat = X_labeled[:, -1, :]

    # 随机抽样
    max_samples = 5000  # 限制样本数以加快计算
    if len(X_flat) > max_samples:
        random_indices = np.random.choice(len(X_flat), max_samples, replace=False)
        X_sample = X_flat[random_indices]
        y_sample = y_labeled[random_indices]
    else:
        X_sample = X_flat
        y_sample = y_labeled

    # 使用PCA进行降维
    try:
        print("执行PCA降维...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sample)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample,
                              cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter)
        plt.title('PCA降维后的特征分布')
        plt.xlabel('主成分1')
        plt.ylabel('主成分2')
        plt.tight_layout()
        plt.savefig('pca_visualization.png')
        print("PCA可视化已保存为 'pca_visualization.png'")

        # 输出主成分方差解释率
        explained_variance = pca.explained_variance_ratio_
        print(f"前两个主成分解释的方差比例: {explained_variance}")
        print(f"累计解释方差比例: {np.sum(explained_variance):.4f}")
    except Exception as e:
        print(f"执行PCA时出错: {e}")

    # 使用t-SNE进行降维
    try:
        print("执行t-SNE降维 (这可能需要几分钟)...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        X_tsne = tsne.fit_transform(X_sample)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample,
                              cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter)
        plt.title('t-SNE降维后的特征分布')
        plt.tight_layout()
        plt.savefig('tsne_visualization.png')
        print("t-SNE可视化已保存为 'tsne_visualization.png'")
    except Exception as e:
        print(f"执行t-SNE时出错: {e}")


def main():
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 加载数据
    X_train, y_train, X_val, y_val, le_fingering = load_data()

    if X_train is None:
        print("数据加载失败，无法进行分析")
        return

    print(f"数据加载完成 - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

    # 分析标签
    analyze_labels(y_train, y_val, le_fingering)

    # 分析特征
    analyze_features(X_train, y_train)

    # 分析特征相关性
    analyze_feature_correlations(X_train, y_train)

    # 可视化高维特征
    visualize_high_dim_features(X_train, y_train)

    print("\n数据分析完成，详细结果请查看生成的图表和CSV文件")
    print("建议在训练前检查以下可能的问题：")
    print("1. 特征中是否存在NaN或Inf值")
    print("2. 标签分布是否严重不平衡")
    print("3. 标注和未标注数据的分布是否一致")
    print("4. 特征与标签的相关性是否有意义")


if __name__ == "__main__":
    main()