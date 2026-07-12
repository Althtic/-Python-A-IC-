import pandas as pd


pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101\alpha_49.csv')


max_idx = data['alpha_49'].idxmin()
start_idx = max_idx - 19

# 3. 处理边界情况：如果最大值在前20行以内，起始索引设为0
if start_idx < 0:
    start_idx = 0
    # 可选：如果前面不足20行，你是想取“从开头到最大值”还是“报错”？
    # 这里默认取从开头到最大值的所有行。
    print(f"提示：最大值位于第 {max_idx} 行，前面不足20行，已返回从第0行到最大值行的数据。")

# 4. 使用 iloc 切片提取数据 (iloc 是左闭右开，所以结束位置要 +1)
# 范围是 [start_idx, max_idx]
result = data.iloc[start_idx : max_idx + 1]

# 打印结果
print(result)

# 如果需要查看最大值的具體信息
print(f"\n最大值: {data['alpha_49'].max()}, 位于索引: {max_idx}")