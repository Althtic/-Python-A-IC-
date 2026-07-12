import pandas as pd
import numpy as np


def process_group_by_date(group, target_factor, layers):
    """
    对一个分组（例如某一天的数据）按指定因子进行分层，并计算每层的平均收益率。

    Args:
        group (pd.DataFrame): 输入的分组数据
        target_factor (str): 用于分层的目标因子列名
        layers (int): 分层数

    Returns:
        pd.DataFrame: 原始数据加上 'quantile' 和 'mean_lndret' 列
    """
    # 1. 先按目标因子值对当前分组进行排序
    sorted_group = group.sort_values(by=target_factor).reset_index(drop=True)

    n = len(sorted_group)

    # 2. 计算基础组大小和余数
    base_size = n // layers  # 每组的基本大小
    remainder = n % layers  # 无法整除后剩下的样本数

    # 3. 创建一个数组来存储每个样本的组号
    # 首先，为每个样本分配一个初步的组号
    # 使用整除运算来分配
    # 这样可以确保前 remainder 个组比后面的组多一个样本
    indices = np.arange(n)
    quantiles = indices // base_size if base_size > 0 else np.zeros(n, dtype=int)

    # 更精确的计算方式，确保前 remainder 个组各多一个元素
    # 计算每个样本应该属于哪个组
    # 第 0 组: [0, base_size + 1)
    # 第 1 组: [base_size + 1, 2*(base_size + 1))
    # ...
    # 第 remainder-1 组: [(remainder-1)*(base_size+1), remainder*(base_size+1))
    # 第 remainder 组: [remainder*(base_size+1), remainder*(base_size+1) + base_size)
    # ...
    # 第 layers-1 组: [...]

    # 一种更简洁的实现方式：
    # 创建一个从 0 开始的数组，长度为 n
    # 让前 remainder * (base_size + 1) 个样本，每 (base_size + 1) 个分为一组
    # 让后面的样本，每 base_size 个分为一组
    quantiles = np.empty(n, dtype=int)
    current_idx = 0
    for i in range(layers):
        size_for_this_group = base_size + (1 if i < remainder else 0)
        quantiles[current_idx: current_idx + size_for_this_group] = i
        current_idx += size_for_this_group

    # 4. 将计算出的分组结果赋值给 DataFrame
    sorted_group['quantile'] = quantiles

    # 5. 计算每组的平均收益率
    sorted_group['mean_lndret'] = sorted_group.groupby('quantile')['holding1D_lndret'].transform('mean')

    # 6. 将结果按原始索引排序，以匹配输入数据的顺序（可选，取决于下游需求）
    # 如果 group 的索引在后续操作中很重要，这一步可以保证顺序一致
    result = sorted_group.reindex(group.index)

    return result

# --- 示例 ---
df_example = pd.DataFrame({
    # 日期列，用于分组（假设按日期分组）
    'date': [
        '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01',
        '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01',

        '2023-01-02', '2023-01-02', '2023-01-02', '2023-01-02',

        '2023-01-03', '2023-01-03', '2023-01-03', '2023-01-03',
        '2023-01-03', '2023-01-03', '2023-01-03'
    ],

    # 股票代码列（或其他标识符）
    'stock': [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R', 'S'
    ],

    # 因子值列，包含了各种情况：
    'factor_value': [
        # 1. 正常分布值
        0.1, 0.5, 0.2, 0.8,
        # 2. 包含重复值
        0.3, 0.3, 0.3, 0.3,
        # 3. 包含 NaN 值
        0.7, 0.9, np.nan, 0.4,
        # 4. 数据量不能被分层数整除 (本例中为7个样本)
        0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    ],

    # 收益率列
    'holding1D_lndret': [
        0.01, -0.02, 0.03, 0.05,
        -0.01, 0.02, -0.03, 0.04,
        0.07, -0.05, 0.01, 0.06,
        -0.01, 0.02, -0.03, 0.04, 0.05, -0.06, 0.07
    ]
})
processed_df = process_group_by_date(df_example, 'factor_value', 5)
print(processed_df)