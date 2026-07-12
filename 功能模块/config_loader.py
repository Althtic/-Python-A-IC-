import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (  # noqa: E402
    BACKTEST_DATA_DIR,
    FACTORS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    backtest_data_path,
    factor_path,
    report_path,
    result_dir,
)
# 因子分析参数
traget_factor = 'alpha_60'      # 分层多空检验的目标因子名称
target_factor = traget_factor
test_window_start = '20190101'  # 回测开始日期
test_window_end = '20251231'    # 回测结束日期
layers = 10                     # 分层层数（默认为5）max = 10
test_period = 5  # 对未来多少个交易日的累计收益率进行IC检测(1个月=21个交易日)
ic_ma_period = 4 * test_period # IC_mean计算窗口长度(持有期整数倍)
holding_period = 5  # 分层回测时持有期长度

# ==================== 配置验证 ====================
def validate_config():
    """配置合法性检查"""
    assert len(traget_factor) > 0, "因子名称不能为空"
    assert 1 <= layers <= 10, "分层层数必须在 1-10 之间"
    assert test_window_start <= test_window_end, "开始日期不能晚于结束日期"
    print("配置验证通过")


if __name__ == '__main__':
    validate_config()
    print(f"当前配置：因子={traget_factor}, 层数={layers}, 窗口={test_window_start}~{test_window_end}")