from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class FrameAudit:
    name: str
    rows: int
    columns: list[str]
    duplicate_key_rows: int | None = None
    null_counts: dict[str, int] | None = None


def read_csv_checked(path: Path, required_columns: Iterable[str] | None = None, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, **kwargs)
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def audit_frame(
    name: str,
    df: pd.DataFrame,
    key_columns: Iterable[str] | None = None,
    null_columns: Iterable[str] | None = None,
) -> FrameAudit:
    duplicate_key_rows = None
    if key_columns:
        duplicate_key_rows = int(df.duplicated(list(key_columns)).sum())

    null_counts = None
    if null_columns:
        null_counts = {col: int(df[col].isna().sum()) for col in null_columns if col in df.columns}

    return FrameAudit(
        name=name,
        rows=len(df),
        columns=list(df.columns),
        duplicate_key_rows=duplicate_key_rows,
        null_counts=null_counts,
    )


def write_audit(path: Path, audits: Iterable[FrameAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(audit) for audit in audits]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

