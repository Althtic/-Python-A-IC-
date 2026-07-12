import tushare as ts
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

pro = ts.pro_api()

def get_industry_name_from_code(index_code):
    """根据申万指数代码返回行业名称（2021版）"""
    mapping = {
        "801010.SI": "农林牧渔",
        "801020.SI": "采掘",
        "801030.SI": "化工",
        "801040.SI": "钢铁",
        "801050.SI": "有色金属",
        "801080.SI": "电子",
        "801110.SI": "家用电器",
        "801120.SI": "食品饮料",
        "801130.SI": "纺织服饰",
        "801140.SI": "轻工制造",
        "801150.SI": "医药生物",
        "801160.SI": "公用事业",
        "801170.SI": "交通运输",
        "801180.SI": "房地产",
        "801200.SI": "商业贸易",
        "801210.SI": "休闲服务",
        "801230.SI": "综合",
        "801710.SI": "建筑材料",
        "801720.SI": "建筑装饰",
        "801730.SI": "电气设备",
        "801740.SI": "国防军工",
        "801750.SI": "计算机",
        "801760.SI": "传媒",
        "801770.SI": "通信",
        "801780.SI": "银行",
        "801790.SI": "非银金融",
        "801880.SI": "汽车",
        "801890.SI": "机械设备"
    }
    return mapping.get(index_code, "未知行业")

def get_sw_industry_mapping(start_date="20170930", end_date="20191231"):
    """
    获取2020-2025年动态申万行业映射表
    返回: {trade_date: {stock_code: industry_name}}
    """
    # 步骤1: 获取所有申万一级行业指数代码
    # 申万一级行业指数代码前缀是801，共28个（21版）

    sw_indices = [
        "801010.SI", "801020.SI", "801030.SI", "801040.SI", "801050.SI",
        "801080.SI", "801110.SI", "801120.SI", "801130.SI", "801140.SI",
        "801150.SI", "801160.SI", "801170.SI", "801180.SI", "801200.SI",
        "801210.SI", "801230.SI", "801710.SI", "801720.SI", "801730.SI",
        "801740.SI", "801750.SI", "801760.SI", "801770.SI", "801780.SI",
        "801790.SI", "801880.SI", "801890.SI"
    ]  # 2021版28个申万一级行业

    # 步骤2: 获取交易日历（A股）
    trade_cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
    trade_dates = trade_cal[trade_cal.is_open == 1]['cal_date'].tolist()

    # print(trade_dates)

    # 步骤3: 构建动态映射表
    dynamic_map = {}
    for date in trade_dates:
        daily_map = {}

        # 重要：Tushare的index_member接口支持按日期查询成分股
        for idx_code in sw_indices:
            try:
                print(f'正在处理{idx_code}在{date}的行业信息')
                # 获取该行业指数在指定日期的成分股
                members = pro.index_member(
                    index_code=idx_code,
                    trade_date=date
                )
                # print(members.head())
                # 提取行业名称（通过指数代码反推）
                industry_name = get_industry_name_from_code(idx_code)
                # print('指数所属行业为：',industry_name)

                # 构建当日映射
                for _, row in members.iterrows():
                    # Tushare返回的ts_code格式：6位数字+交易所后缀
                    stock_code = row['con_code']
                    daily_map[stock_code] = industry_name
                time.sleep(0.3)
                # print(daily_map)

            except Exception as e:
                # 处理接口限流/异常
                print(f"Error at {date} for {idx_code}: {str(e)}")
                continue

        dynamic_map[date] = daily_map
        print(f"Processed {date}, found {len(daily_map)} stocks")
    print("处理完毕")
    return dynamic_map

processed_dynamic_map = get_sw_industry_mapping()
# print(processed_dynamic_map['20200102'])
print(processed_dynamic_map)
# 辅助函数：申万指数代码转行业名称



# --- 主程序 ---


# --- 将字典转换为DataFrame ---
# 1. 创建一个 Series，索引是日期，值是股票-行业字典
temp_series = pd.Series(processed_dynamic_map, name='stock_industry_dict')
df_temp = temp_series.reset_index()
df_temp.columns = ['trade_date', 'stock_industry_dict']
print("Step 1: Series to DataFrame with dict column")
print(df_temp)
print("\n" + "="*50 + "\n")

# 2. 使用 explode 将字典展开成多行
df_exploded = df_temp.explode('stock_industry_dict')
print("Step 2: Explode dict column into multiple rows")
print(df_exploded)
print("\n" + "="*50 + "\n")

# 3. 从 exploded 的 Series ('stock_industry_dict') 中提取 key 和 value
# 创建一个临时的 DataFrame，其中 index 是股票代码，value 是行业名称
# 然后 reset_index 将 index 变成一列，从而获得股票代码列
temp_key_val_df = df_exploded['stock_industry_dict'].apply(lambda x: pd.Series(x)).reset_index(drop=True)
print('----------------------------------------')
print(temp_key_val_df)
# 由于字典只有一个键值对，我们直接取它的 index (股票代码) 和 value (行业名称)
# 但 explode 已经将字典展开成了 Series，Series 的 index 就是股票代码，value 就是行业名称
# 所以我们需要重新构建一个 DataFrame

# 更简单的方法：直接利用 explode 后的 Series 的 index 和 value
# explode 之后，Series 的 index 是原来的索引（对应 trade_date），value 是字典
# 但我们想要的是：对于每个字典，展开其内部的 key-value 对

# 重新考虑：explode 后，每一行的 'stock_industry_dict' 是一个 dict
# 我们需要将这个 dict 展开为两列，这正是 apply(pd.Series) 的作用
# 为了使用 melt，我们需要先将这些 dict 变成一个宽格式 DataFrame

# 这个思路有点绕。让我们用一个更直接的方法来用 melt：

# 重新开始，构建一个中间的宽格式 DataFrame 来使用 melt
all_rows_for_melt = []
for trade_date, stock_industry_dict in processed_dynamic_map.items():
    if stock_industry_dict:  # 确保字典不为空
        # 创建一行，其中列名是股票代码，值是行业名称
        row_data = {'trade_date': trade_date}
        row_data.update(stock_industry_dict) # 将股票代码:行业名 添加为列
        all_rows_for_melt.append(row_data)

# 创建宽格式 DataFrame
wide_df = pd.DataFrame(all_rows_for_melt)
print("Intermediate Wide DataFrame for Melt:")
print(wide_df)
print("\n" + "="*50 + "\n")

# 3. 使用 melt 将宽格式 DataFrame 转换为长格式
# id_vars: 保持不变的列 (trade_date)
# value_vars: 需要熔化的列 (股票代码列)
# var_name: 新的变量列名 (stock_code)
# value_name: 新的值列名 (industry_name)

if not wide_df.empty:
    # 获取所有列名，并排除 'trade_date' 作为要熔化的列
    id_vars = ['trade_date']
    value_vars = [col for col in wide_df.columns if col != 'trade_date']

    long_df = wide_df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='stock_code',
        value_name='industry_name'
    )

    # 4. 清理数据：删除 industry_name 为 NaN 的行
    final_df_melt = long_df.dropna(subset=['industry_name'])

    # 5. 按照 trade_date 降序排列 (最近的日期在前)
    final_df_melt_sorted = final_df_melt.sort_values(by='trade_date', ascending=True).reset_index(drop=True)

    print("Final DataFrame using Melt, sorted by trade_date descending:")
    print(final_df_melt_sorted)
else:
    print("No data to melt.")

final_df_melt_sorted.to_csv("SWlevel1_sorted_20170930-20191231.csv")
print('over!')