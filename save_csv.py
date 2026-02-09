import os
import logging
import pandas as pd  # 需要导入 pandas，因为函数内部用了 df.to_csv()

# --- 配置日志 ---
# level=logging.INFO 表示记录 INFO 及以上级别的信息
# format 定义了日志消息的格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

def save_data(df, output_filename, base_dir=None):
    """
    将包含历史数据的DataFrame保存到指定目录下的CSV文件。
    如果文件已存在，将被覆盖。

    Parameters:
    df (pd.DataFrame): 要保存的DataFrame。
    output_filename (str): 要保存的CSV文件名 (例如 "Alpha#01.csv")。
    base_dir (str, optional): 基础输出目录。如果未提供，则默认为脚本所在目录下的 'WorldQuant_Alpha101' 子目录。
                              例如: "C:\\path\\to\\your\\custom\\dir"
    """
    if base_dir is None:
        base_dir = r"C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101"
    # 构建完整的输出路径
    full_output_path = os.path.join(base_dir, output_filename)
    # 检查目录是否存在，如果不存在则创建
    os.makedirs(base_dir, exist_ok=True)
    # 检查文件是否存在
    if os.path.exists(full_output_path):
        logger.info(f"提示: 文件 '{full_output_path}' 已存在，旧文件将被覆盖。")
    try:
        # 将 DataFrame 保存为 CSV 文件 (默认行为就是覆盖)
        df.to_csv(full_output_path, index=False)
        logger.info(f"数据已成功保存/覆盖至: {full_output_path}")
    except PermissionError as e:
        logger.info(f"错误: 没有权限写入文件 '{full_output_path}'. 详细信息: {e}")
    except OSError as e:  # 捕获其他可能的系统级 I/O 错误
        logger.info(f"错误: 写入文件 '{full_output_path}' 时发生系统错误. 详细信息: {e}")
    except Exception as e:  # 捕获其他未预期的异常
        logger.info(f"错误: 保存文件 '{full_output_path}' 时发生未知错误. 详细信息: {e}")

