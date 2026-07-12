# QuantSystem

A-share factor research prototype based on Tushare-style OHLC, fundamental and industry data.

The current reproducible main path focuses on `alpha_60`: factor calculation, IC validation, quantile spread backtest, FF5 regression and optional LLM report generation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Layout

This repository intentionally ignores large local datasets. Put your data under the project root using these relative paths:

```text
回测数据集/
  20170930-20251231_pipe.csv
  rf.csv
Factors/
  FF5.csv
```

Generated factor files are written to `Factors/`. Validation results are written to `因子检验结果/<factor>/`. LLM reports are written to `因子分析报告/`.

## Run

Run the default pipeline:

```powershell
python run_pipeline.py
```

Run selected steps:

```powershell
python run_pipeline.py --steps validation quantile regression
```

Generate the optional DeepSeek report after setting an API key:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
python run_pipeline.py --steps report
```

## Notes

- Runtime paths are resolved from the cloned project directory via `project_paths.py`.
- Large CSV datasets, generated plots, caches and IDE files are ignored by `.gitignore`.
- `功能模块/config_loader.py` still uses the historical name `traget_factor` for compatibility; `target_factor` is provided as an alias.

