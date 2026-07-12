import pandas as pd

# 1. 指定你要合并的文件路径列表，并按你想要的顺序排列
file_list = [
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20190331.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20190630.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20190930.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20191231.xlsx',

    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20200331.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20200630.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20200930.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20201231.xlsx',

    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20210331.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20210630.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20210930.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20211231.xlsx',

    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20220331.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20220630.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20220930.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20221231.xlsx',

    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20230331.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20230630.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20230930.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20231231.xlsx',

    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20240331.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20240630.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20240930.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20241231.xlsx',

    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20250331.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20250630.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20250930.xlsx',
    r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务数据\20251231.xlsx',

    # ... 添加更多文件路径
]

# 2. 读取并合并
dfs = []
for file_path in file_list:
    print(f"正在读取: {file_path}")
    df = pd.read_excel(file_path) # 如果有特定的sheet_name，可以加上
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True, sort=False)

# 3. 保存为CSV
output_csv_path = r'19-25_merged_output.csv' # 替换成你的输出路径
combined_df.to_csv(output_csv_path, index=False)

print(f"合并完成！文件已保存至: {output_csv_path}")

# # 4. 保存为excel
# output_excel_path = r'19-25_merged_output.xlsx' # 替换成你的输出路径
# combined_df.to_excel(output_excel_path, index=False)
#
# print(f"合并完成！文件已保存至: {output_excel_path}")