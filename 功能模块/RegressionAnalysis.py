from unittest import result
import warnings
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from config_loader import traget_factor, test_window_start, test_window_end

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def data_loading(traget_factor):
    logger.info(f"目标检测因子: {traget_factor}")
    base_directory = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors'
    alpha = traget_factor
    filename = f"{alpha}.csv"
    Input_path = base_directory + '\\' + filename
    df_loading = pd.read_csv(Input_path)
    return df_loading

def cut_time_window(df, test_window_start, test_window_end):
    try:
        if df['trade_date'].dtype in ['int64', 'int32']:
            df_date_for_check = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
        elif df['trade_date'].dtype == 'object':
            df_date_for_check = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        else:
            df_date_for_check = df['trade_date']

        original_min_date = df_date_for_check.min()
        original_max_date = df_date_for_check.max()
        logger.info(f"原始数据的时间范围: {original_min_date.strftime('%Y%m%d')} 到 {original_max_date.strftime('%Y%m%d')}")

        input_start_dt = pd.to_datetime(str(test_window_start), format='%Y%m%d')
        input_end_dt = pd.to_datetime(str(test_window_end), format='%Y%m%d')

        if input_start_dt > original_max_date or input_end_dt < original_min_date:
            logger.info(f"错误: 请求的时间窗口 [{test_window_start}, {test_window_end}] 完全超出了原始数据的日期范围 [{original_min_date.strftime('%Y%m%d')}, {original_max_date.strftime('%Y%m%d')}]。")
            logger.info("--- 程序终止 ---")
            return None 

        if input_start_dt < original_min_date or input_end_dt > original_max_date:
             effective_start = max(input_start_dt, original_min_date)
             effective_end = min(input_end_dt, original_max_date)
             logger.info(f"警告: 请求的时间窗口 [{test_window_start}, {test_window_end}] 部分超出了原始数据范围。")
             logger.info(f"     建议使用有效范围: [{effective_start.strftime('%Y%m%d')}, {effective_end.strftime('%Y%m%d')}]")
             return None 

        df_copy = df.copy()
        if df_copy['trade_date'].dtype in ['int64', 'int32']:
            df_copy['trade_date'] = df_copy['trade_date'].astype(str)

        mask = (df_copy['trade_date'] >= test_window_start) & (df_copy['trade_date'] <= test_window_end)
        df_in_window = df_copy.loc[mask]
        df_in_window = df_in_window.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
        return df_in_window

    except Exception as e:
        logger.info(f"处理时间窗口时发生错误: {e}")
        return None

def data_preprocessing(df, traget_factor, test_window_start, test_window_end):
    df = df.copy()
    df = cut_time_window(df, test_window_start, test_window_end)
    rf = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\rf.csv')
    rf_filtered = rf[['trade_date', 'rf']].copy()
    rf_filtered['trade_date'] = rf_filtered['trade_date'].astype(str)
    df_rf = df.merge(rf_filtered, on=['trade_date'], how='left')
    df_rf['excess_dret'] = df_rf['dret'] - df_rf['rf']
    return df_rf

def regression_analysis_by_date(df, factor_col, target_col='excess_dret'):
    results = []
    grouped = df.groupby('trade_date')
  
    total_groups = len(grouped)
    logger.info(f"开始进行 {total_groups} 个交易日的截面回归分析")
    for date, group in grouped:
        if isinstance(factor_col, list):
            cols_to_check = [target_col] + factor_col
        else:
            cols_to_check = [target_col, factor_col]
        subset = group.dropna(subset=cols_to_check)
       
        n_samples = len(subset)
        if n_samples < 10:
            logger.info(f"日期 {date} 有效样本不足，跳过回归分析")
            continue
        y = subset[target_col].values
        X = subset[factor_col].values
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model = LinearRegression(fit_intercept=True)
        try:
            model.fit(X, y)
                
            alpha_val = float(model.intercept_) 
            # model.coef_ 是数组，单因子时形状为 (1,)，取第一个元素转为 float
            beta_val = float(model.coef_[0]) 
            r_squared_val = float(model.score(X, y))
            res_row = {
                'trade_date': date,
                'alpha': alpha_val,      
                'beta': beta_val,      
                'r_squared': r_squared_val
            }
            results.append(res_row)
            
        except Exception as e:
            logger.info(f"日期 {date} 回归分析失败: {e}")
            continue
    
    result_df = pd.DataFrame(results)
    logger.info(f"回归分析完成, 正确处理 {len(result_df)} 个交易日")
    return result_df

def beta_validation_test(result_df):
    beta_s = result_df['beta'].dropna()
    if len(beta_s) < 2:
        return None
    T = len(beta_s)
    maxlags = max(1, int(4 * (T / 100) ** (2 / 9)))
    X = np.ones((T, 1))
    model = OLS(beta_s.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    fm_t = float(np.abs(model.tvalues[0]))
    fm_se = float(model.bse[0])
    fm_mean_beta = float(model.params[0])
    pass_196 = fm_t > 1.96
    pass_3 = fm_t > 3
    win_rate = float((beta_s > 0).mean())
    kurt = float(kurtosis(beta_s, nan_policy='omit'))
    skew_val = float(skew(beta_s, nan_policy='omit'))
    fb_results = {
        'fm_mean_beta': fm_mean_beta,
        'fm_nw_se': fm_se,
        'fm_t_abs': fm_t,
        'pass_196': pass_196,
        'pass_3': pass_3,
        'beta_win_rate': win_rate,
        'beta_kurtosis': kurt,
        'beta_skewness': skew_val,
    }
    return fb_results

def print_beta_results_native(results):
    """使用原生字符串格式化打印结果"""
    if results is None:
        print("验证失败：样本量不足 (N < 2)")
        return

    # 辅助函数：格式化布尔值
    def fmt_bool(val):
        return "Yes" if val else "No"

    print("\n========================================")
    print(f"{'均值 Beta':<20}  {results['fm_mean_beta']:>12.4f}")
    print(f"{'Newey-West SE':<20}  {results['fm_nw_se']:>12.4f}")
    print(f"{'T统计量 (Abs)':<20}  {results['fm_t_abs']:>12.4f}")
    print(f"{'显著性 (|t|>1.96)':<20}  {fmt_bool(results['pass_196']):>12}")
    print(f"{'强显著 (|t|>3.0)':<20}  {fmt_bool(results['pass_3']):>12}")
    print(f"{'胜率 (Win Rate)':<20}  {results['beta_win_rate']:>11.2%}")
    print(f"{'偏度 (Skewness)':<20}  {results['beta_skewness']:>12.4f}")
    print(f"{'峰度 (Kurtosis)':<20}  {results['beta_kurtosis']:>12.4f}")
    print("========================================\n")


def run():
    df = data_loading(traget_factor)
    df = data_preprocessing(df, traget_factor, test_window_start, test_window_end)
    result_df = regression_analysis_by_date(df, factor_col=traget_factor)
    fb_results = beta_validation_test(result_df)
    print_beta_results_native(fb_results)




if __name__ == "__main__":
    run()