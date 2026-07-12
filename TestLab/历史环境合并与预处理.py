import pandas as pd
import os
import glob

def merge_stock_data_interactive():
    """交互式版本，直接要求用户输入日期"""

    print("股票数据合并工具")
    print("=" * 50)

    # 获取用户输入
    start_date = input("请输入开始日期 (格式: YYYYMMDD 或 YYYY-MM-DD): ").strip()
    end_date = input("请输入结束日期 (格式: YYYYMMDD 或 YYYY-MM-DD): ").strip()

    # 移除日期中的连字符
    start_date_clean = start_date.replace('-', '')
    end_date_clean = end_date.replace('-', '')

    # 验证日期格式
    if len(start_date_clean) != 8 or not start_date_clean.isdigit():
        print("错误：开始日期格式不正确，请使用8位数字格式")
        return

    if len(end_date_clean) != 8 or not end_date_clean.isdigit():
        print("错误：结束日期格式不正确，请使用8位数字格式")
        return

    if start_date_clean > end_date_clean:
        print("错误：开始日期不能晚于结束日期")
        return

    # 数据文件夹路径
    data_folder = r"C:\Users\63585\Desktop\PycharmProjects\pythonProject\StockDailyData"

    # 检查文件夹是否存在
    if not os.path.exists(data_folder):
        print(f"错误：数据文件夹 '{data_folder}' 不存在")
        return

    print(f"\n正在处理 {start_date_clean} 到 {end_date_clean} 的数据...")

    # 使用通配符匹配文件
    file_pattern = os.path.join(data_folder, f"daily_stock_*.csv")
    all_csv_files = glob.glob(file_pattern)

    if not all_csv_files:
        print(f"错误：在文件夹 '{data_folder}' 中没有找到任何CSV文件")
        return

    # 筛选在日期范围内的文件
    filtered_files = []
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        # 提取日期部分
        date_str = filename.replace('daily_stock_', '').replace('.csv', '')

        if start_date_clean <= date_str <= end_date_clean:
            filtered_files.append(file_path)

    if not filtered_files:
        print(f"在 {start_date_clean} 到 {end_date_clean} 范围内没有找到数据文件")
        return

    print(f"找到 {len(filtered_files)} 个文件，正在合并...")

    # 读取并合并所有文件
    dfs = []
    file_count = len(filtered_files)

    for i, file_path in enumerate(filtered_files, 1):
        try:
            print(f"正在读取文件 {i}/{file_count}: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"读取 {os.path.basename(file_path)} 失败: {e}")

    if not dfs:
        print("没有成功读取任何文件")
        return

    print("正在合并数据...")
    merged_df = pd.concat(dfs, ignore_index=True)

    # 保存结果
    output_file = f"daily_stock_{start_date_clean}_{end_date_clean}.csv"
    merged_df.to_csv(output_file, index=False)

    print("\n" + "=" * 50)
    print("合并完成！")
    print(f"数据已保存到: {output_file}")
    print(f"总数据量: {len(merged_df)} 行")
    print(f"数据列: {', '.join(merged_df.columns.tolist())}")

    # 显示数据预览
    print("\n数据预览:")
    print(merged_df.head())

    return merged_df


# 主程序入口
if __name__ == "__main__":
    # 直接运行交互式函数
    merge_stock_data_interactive()

    # 可选：询问用户是否继续
    while True:
        choice = input("\n是否继续合并其他日期范围的数据？(y/n): ").strip().lower()
        if choice == 'y' or choice == 'yes':
            merge_stock_data_interactive()
        else:
            print("程序结束")
            break