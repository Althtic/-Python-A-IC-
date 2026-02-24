import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def distribution_plot(df):
    """绘制最后一列的直方图，并显示基础统计指标"""
    last_col = df.columns[-1]
    data = df[last_col].dropna()

    # ===== 计算统计指标 =====
    mean_val = data.mean()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    median_val = data.median()
    skew_val = data.skew()
    kurt_val = data.kurt()
    count_val = len(data)

    # ===== 创建图形 =====
    fig, ax = plt.subplots(figsize=(10, 6))

    # ===== 绘制直方图 =====
    n, bins, patches = ax.hist(data, bins=30, edgecolor='black',
                               color='skyblue', alpha=0.7, label='Distribution')

    # ===== 添加均值线（红色虚线） =====
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')

    # ===== 添加中位数线（绿色虚线） =====
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')

    # ===== 添加±1标准差范围（阴影区域） =====
    ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='gray',
               label=f'±1 Std: [{mean_val - std_val:.4f}, {mean_val + std_val:.4f}]')

    # ===== 添加统计指标文本框 =====
    stats_text = (
        f'Statistics Summary\n'
        f'─────────────────\n'
        f'Count:   {count_val:>10,}\n'
        f'Mean:    {mean_val:>10.4f}\n'
        f'Std:     {std_val:>10.4f}\n'
        f'Min:     {min_val:>10.4f}\n'
        f'25%:     {data.quantile(0.25):>10.4f}\n'
        f'Median:  {median_val:>10.4f}\n'
        f'75%:     {data.quantile(0.75):>10.4f}\n'
        f'Max:     {max_val:>10.4f}\n'
        f'Skew:    {skew_val:>10.4f}\n'
        f'Kurt:    {kurt_val:>10.4f}'
    )

    # 在图右上角添加文本框
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            fontfamily='monospace')

    # ===== 设置标题和标签 =====
    ax.set_title(f'Distribution of {last_col}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # ===== 添加图例 =====
    ax.legend(loc='upper left', fontsize=9)

    # ===== 调整布局并显示 =====
    plt.tight_layout()
    plt.show()

    # ===== 同时打印到控制台（方便复制） =====
    print(f"\n{'=' * 50}")
    print(f"Distribution Summary: {last_col}")
    print(f"{'=' * 50}")
    print(f"Count:   {count_val:>10,}")
    print(f"Mean:    {mean_val:>10.4f}")
    print(f"Std:     {std_val:>10.4f}")
    print(f"Min:     {min_val:>10.4f}")
    print(f"25%:     {data.quantile(0.25):>10.4f}")
    print(f"Median:  {median_val:>10.4f}")
    print(f"75%:     {data.quantile(0.75):>10.4f}")
    print(f"Max:     {max_val:>10.4f}")
    print(f"Skew:    {skew_val:>10.4f}")
    print(f"Kurt:    {kurt_val:>10.4f}")
    print(f"{'=' * 50}\n")

    return {
        'count': count_val,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'median': median_val,
        'skew': skew_val,
        'kurt': kurt_val
    }