# QuantSystem / WorldQuant101 in A-shares

QuantSystem 是一个面向 A 股的量化因子研究系统。项目从本地 CSV 原型演进为以 DolphinDB 为数据底座、Python 为计算层、FastAPI + 静态前端为研究工作台的完整流程，用于管理行情、财务、行业、股票截面 Alpha 因子和 FF5 市场风险因子。

当前主线能力：

- 将日频行情、停复牌状态、估值、申万一级行业和基础财务特征写入 `dfs://quant_system`。
- 计算并入库多只 WorldQuant 风格 Alpha 因子，结果写入 `factor_daily`。
- 计算 FF5 市场风险因子 `mkt/smb/hml/rmw/cma`，结果写入 `market_factor_daily`，无风险利率写入 `macro_rate_daily`。
- 提供 IC 检验、分层回测、行业内分层、市场因子看板等 API 和网页展示。
- 保留 CSV 版历史脚本，用于 legacy 复现、基准修正和 DB/CSV 一致性检查。

## 项目结构

```text
QuantSystem/
  data_pipeline/              # DolphinDB 同步、Tushare 更新、分析数据读取
  dolphindb_scripts/          # DolphinDB schema 和示例导入脚本
  Factor_Calculate/           # WQ Alpha 与 FF5 CSV 版公式实现
  功能模块/                   # IC、分层回测、回归分析等历史研究模块
  web_backend/                # FastAPI 后端
  web_frontend/               # 静态前端页面、图表和表格
  docs/                       # 迁移说明、数据血缘、基准和一致性报告
  tests/                      # 单元测试
  ff5_core.py                 # CSV/DB 共用 FF5 2x3 计算核心
  run_alpha_dolphindb.py      # Alpha 因子计算入库入口
  run_ff5_dolphindb.py        # FF5/RF 入库、计算、比对和检查入口
  run_validation_dolphindb.py # IC 检验 CLI
  run_quantile_dolphindb.py   # 分层回测 CLI
  run_research_system.py      # FastAPI + 静态前端启动入口
```

## 环境准备

建议使用 Python 3.10+。克隆后在项目根目录安装依赖：

```powershell
git clone https://github.com/Althtic/WorldQuant101_in_Ashares.git
cd WorldQuant101_in_Ashares
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

本项目默认连接本机 DolphinDB：

```text
host: 127.0.0.1
port: 8848
user: admin
password: 123456
database: dfs://quant_system
```

如果使用 Tushare 增量更新，需要在 `.env` 或环境变量中配置 token。`.env`、大体量 CSV、日志和生成报告不会提交到仓库。

## 数据准备

仓库不包含大体量研究数据。可按以下相对路径放置本地 CSV 数据，用于历史复现或首次导入：

```text
回测数据集/
  20170930-20251231_pipe.csv
  rf.csv
Factors/
  FF5.csv
```

推荐主线是 DolphinDB。首次建库和同步请先启动 DolphinDB，再运行：

```powershell
python run_dolphindb_sync.py
```

Tushare 增量更新入口：

```powershell
python run_tushare_update.py market --lookback-trading-days 5
python run_tushare_update.py all --start-date 20250101 --end-date 20251231
```

## 运行研究工作台

启动后端和静态前端：

```powershell
python run_research_system.py
```

访问：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health | Select-Object -ExpandProperty Content
```

主要 API：

| API | 说明 |
| --- | --- |
| `GET /api/health` | DolphinDB 连接检查 |
| `GET /api/tables/status` | 核心表行数和日期范围 |
| `GET /api/factors` | 股票截面 Alpha 因子列表 |
| `GET /api/market-factors` | FF5 等市场因子列表 |
| `GET /api/factors/{factor_name}/ic` | 因子 IC 检验 |
| `GET /api/factors/{factor_name}/quantile` | 因子分层回测 |

## Alpha 因子计算

通用 Alpha runner 从 DolphinDB 读取行情、状态、市值和行业数据，调用 `Factor_Calculate/WQ_Alpha*.py` 中的公式，统一完成复牌窗口过滤、去极值、市值/行业中性化，并写回 `factor_daily`。

短窗口测试：

```powershell
python -B run_alpha_dolphindb.py alpha_07 --start-month 2019-01 --end-month 2019-01 --replace-window
```

单因子全量：

```powershell
python -B run_alpha_dolphindb.py alpha_37 --replace
```

已验证因子批量更新：

```powershell
python -B run_alpha_dolphindb.py scheduled --replace-window
```

当前通用 runner 覆盖 `alpha_01`、`alpha_02`、`alpha_03`、`alpha_04`、`alpha_05`、`alpha_06`、`alpha_07`、`alpha_08`、`alpha_09`、`alpha_11`、`alpha_12`、`alpha_13`、`alpha_14`、`alpha_15`、`alpha_16`、`alpha_17`、`alpha_18`、`alpha_23`、`alpha_25`、`alpha_26`、`alpha_27`、`alpha_28`、`alpha_30`、`alpha_32`、`alpha_33`、`alpha_34`、`alpha_35`、`alpha_37`、`alpha_45`、`alpha_49`、`alpha_50`、`alpha_61`。

`alpha_57` 和 `alpha_60` 保留专用脚本：

```powershell
python -B run_alpha57_dolphindb.py --replace --factor-version v1
python -B run_alpha60_dolphindb.py --replace --factor-version v1
```

## FF5 市场因子

FF5 是市场级时间序列，不带 `ts_code` 维度，因此单独写入 `market_factor_daily`，不要混入股票截面 Alpha 的 `factor_daily`。

导入 CSV 版 FF5：

```powershell
python -B run_ff5_dolphindb.py import-csv --path Factors/FF5.csv --replace-window
```

更新 Tushare 无风险利率：

```powershell
python -B run_ff5_dolphindb.py update-rf --start-date 20171010 --end-date 20251231
```

按 DolphinDB 数据流式计算 FF5：

```powershell
python -B run_ff5_dolphindb.py compute `
  --start-date 20171010 --end-date 20251231 `
  --chunk-months 1 `
  --factor-version v2 `
  --data-version ff5_db_stream_v2 `
  --replace-window
```

DB/CSV 一致性检查：

```powershell
python -B run_ff5_dolphindb.py compare-csv `
  --path Factors/FF5_fixed_20260711.csv `
  --start-date 20171010 --end-date 20251231 `
  --chunk-months 1 `
  --output docs\ff5_db_vs_csv_v2_20171010_20251231.json

python -B run_ff5_dolphindb.py check --start-date 20171010 --end-date 20251231 --factor-version v2
```

## IC 检验与分层回测

IC 检验：

```powershell
python -B run_validation_dolphindb.py --factor-name alpha_60 --factor-version v1 --start-date 20190101 --end-date 20190331
```

分层回测：

```powershell
python -B run_quantile_dolphindb.py --factor-name alpha_60 --factor-version v1 --start-date 20190101 --end-date 20190331
```

API 默认使用 DolphinDB 加速路径；`analysis_engine=python` 可用于历史逻辑对照。前端展示文案以中文为主，`Rank IC` 保留英文。

## 测试与检查

不依赖 DolphinDB 的本地检查：

```powershell
python -B -m pytest tests/test_ff5_core.py tests/test_ic_drawdown.py tests/test_tushare_update.py
node --check web_frontend\app.js
python -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['web_backend/api.py','run_alpha_dolphindb.py','run_ff5_dolphindb.py','ff5_core.py']]"
```

需要 DolphinDB 的检查：

```powershell
python -B run_ff5_dolphindb.py check --start-date 20171010 --end-date 20251231 --factor-version v2
```

## 仓库维护说明

- 大型 CSV、生成图表、运行日志、`.env`、IDE 配置和 Python 缓存默认忽略。
- `docs/*.json` 中保留了小型基准报告，便于理解迁移和一致性状态；`docs/*.csv` 为生成明细，默认忽略。
- 中文路径是历史项目结构的一部分，Windows + PowerShell 下请优先使用 UTF-8 终端显示。
- `功能模块/config_loader.py` 和部分历史模块保留旧拼写 `traget_factor` 兼容逻辑，不要贸然全局改名。
