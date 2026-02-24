import pandas as pd

def remove_resume_window_data(df, window, resume_col='suspend_type'):
    """
    删除复牌日之后 window 长度的数据

    参数:
        df: 因子计算后的数据（包含 ST_type 列）
        window: 滚动窗口大小
        resume_col: 复牌日标记列名
    """
    df = df.copy()
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 标记复牌日
    df['is_resume'] = df[resume_col] == 'R'
    # 对复牌日分组并计数（核心逻辑）
    df['resume_group'] = df['is_resume'].groupby(df['ts_code']).cumsum()
    df['days_after_resume'] = df.groupby(['ts_code', 'resume_group']).cumcount()
    # 复牌日及之后 window-1 天标记为删除（共 window 天）
    df['to_drop'] = (df['resume_group'] > 0) & (df['days_after_resume'] < window)
    # print(df[df['ts_code'] == '000004.SZ'])
    # 删除标记的行
    df_clean = df[~df['to_drop']].copy()
    # print(df[df['ts_code'] == '000004.SZ'])
    # 清理临时列
    df_clean = df_clean.drop(columns=['is_resume', 'resume_group', 'days_after_resume', 'to_drop'])
    df_clean = df_clean.reset_index(drop=True)

    print(f"原始数据量：{len(df):,}")
    print(f"删除复牌窗口数据：{(df['to_drop']).sum():,}")
    print(f"清洗后数据量：{len(df_clean):,}")

    return df_clean
