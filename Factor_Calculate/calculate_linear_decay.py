import numpy as np
from numba import njit

def linear_decay_peaks(series, codes, window):
    """
    对按 `codes` 分组的时间序列做“窗口内峰值线性衰减加权和”。

    计算逻辑（每个分组内独立进行）：
    - 对分组内每个位置 i，从该分组的第 `window-1` 个点开始计算；窗口为 [i-window+1, i]（含端点）。
    - 在窗口内逐点判断是否为“峰值”：
      - 窗口内部点 j：`values[j] >= values[j-1]` 且 `values[j] >= values[j+1]`
      - 窗口边界点：只与窗口内可用邻点比较（窗口只有 1 个点时视为峰值）
    - 仅对峰值点累加：`sum(values[j] * w)`，其中线性权重 `w = (k+1)/window`，k 为窗口内相对位置
      （k=0 为最旧点，k=window-1 为最新点，最新点权重最大）。

    参数
    - series: array-like (N,)
      数值序列（如某个因子/价格/成交量衍生值）。会被转换为 float64。
    - codes: array-like (N,)
      分组标识（如股票代码）。要求与 `series` 等长，且**同一 code 的样本在数组中必须是连续段**
      （通常意味着你已按 `codes` + 时间排序）。算法用相邻变化点来切分分组。
    - window: int
      窗口长度，需为正整数。若某分组长度 < window，则该分组全部输出为 NaN。

    返回
    - out: np.ndarray (N,), float64
      与输入等长。每个分组内：前 `window-1` 个位置为 NaN；其余位置为窗口内峰值的线性加权和。

    示例
    - 假设 `df` 已按 `['ts_code','trade_date']` 升序排列：
      `df['x_decay_peak'] = linear_decay_peaks(df['x'].to_numpy(), df['ts_code'].to_numpy(), window=10)`
    """
    values = np.asarray(series, dtype=np.float64)
    # 数组切片错位比较来找到分组变化的位置
    change = np.where(codes[:-1] != codes[1:])[0] + 1
    group_starts = np.concatenate(([0], change))
    group_lengths = np.diff(np.concatenate((group_starts, [len(codes)])))
    
    return calc_grouped_linear_decay_peaks(values, group_starts, group_lengths, window)

@njit
def _is_peak(values, j, start, end):
    if j <= start:
        return values[j] >= values[j + 1] if j + 1 <= end else True
    if j >= end:
        return values[j] >= values[j - 1] if j - 1 >= start else True
    return values[j] >= values[j - 1] and values[j] >= values[j + 1]

@njit
def calc_grouped_linear_decay_peaks(values, group_starts, group_lengths, window):
    """
    `linear_decay_peaks` 的 numba 加速核心实现。

    参数
    - values: np.ndarray (N,), float64
    - group_starts: np.ndarray (G,), int
      每个分组在 `values` 中的起始下标。
    - group_lengths: np.ndarray (G,), int
      每个分组长度；与 `group_starts` 一一对应。
    - window: int

    返回
    - out: np.ndarray (N,), float64
      规则同 `linear_decay_peaks`。
    """
    n_total = len(values)
    out = np.full(n_total, np.nan)
    for g_idx in range(len(group_starts)):
        start = group_starts[g_idx]
        length = group_lengths[g_idx]
        end = start + length
        if length < window:
            continue
        for i in range(start + window - 1, end):
            w_start = i - window + 1
            s = 0.0
            for k in range(window):
                j = w_start + k
                if not _is_peak(values, j, w_start, i):
                    continue
                w = (k + 1) / window
                s += values[j] * w
            out[i] = s
    return out
