import numpy as np
from numba import njit

@njit
def rolling_corr_numba(target_array, feature_array, window):
    n = len(target_array)
    result = np.full(n, np.nan)  # 初始化结果数组，避免后续拼接错乱
    # print(target_array[:5])
    # print('----------------')
    # print(feature_array[:5])
    # print('----------------')
    for i in range(window, n):  # 从第 window 个元素开始计算
        # 提取窗口内的子数组
        tgt_win = target_array[i - window:i]
        ftr_win = feature_array[i - window:i]

        # 计算均值
        mean_tgt = np.mean(tgt_win)
        mean_ftr = np.mean(ftr_win)

        # 计算协方差和方差 (手动计算以避免 np.cov 的开销)
        cov = 0.0
        var_tgt = 0.0
        var_ftr = 0.0
        for j in range(window):
            cov += (tgt_win[j] - mean_tgt) * (ftr_win[j] - mean_ftr)
            var_tgt += (tgt_win[j] - mean_tgt) ** 2
            var_ftr += (ftr_win[j] - mean_ftr) ** 2

        # 防止除0
        if var_tgt > 1e-8 and var_ftr > 1e-8:
            corr = cov / (np.sqrt(var_tgt) * np.sqrt(var_ftr))
        else:
            corr = np.nan
        result[i] = corr
    return result