from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class OverviewBuildConfig:
    input_dir: str
    out_csv: str
    file_pattern: str = "*-OverviewSpectra.csv"
    mz_col: str = "m/z"
    intensity_col: str = "intensities"
    align_how: str = "inner"
    label_from_filename: bool = True
    labels: Optional[Sequence[str]] = None


def _read_scils_semicolon_with_comments(path: str) -> Tuple[List[str], pd.DataFrame]:
    comments: List[str] = []
    data_lines: List[str] = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line.rstrip("\n"))
            elif line.strip():
                data_lines.append(line)

    if not data_lines:
        raise ValueError(f"{path} 中没有表格数据。")

    df = pd.read_csv(StringIO("".join(data_lines)), sep=";")
    return comments, df


def read_overview_spectrum(path: str, mz_col: str = "m/z", intensity_col: str = "intensities") -> Tuple[List[str], pd.DataFrame]:
    comments, df = _read_scils_semicolon_with_comments(path)
    if mz_col not in df.columns or intensity_col not in df.columns:
        raise ValueError(
            f"{path} 缺少必需列。当前列为 {list(df.columns)}，需要包含 {mz_col!r} 和 {intensity_col!r}。"
        )

    out = df[[mz_col, intensity_col]].copy()
    out.columns = ["mz", "intensity"]
    out["mz"] = pd.to_numeric(out["mz"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    out = out.dropna(subset=["mz"]).sort_values("mz").reset_index(drop=True)
    out["intensity"] = out["intensity"].fillna(0.0)
    return comments, out


def infer_label_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("-OverviewSpectra"):
        return stem[:-len("-OverviewSpectra")]
    return stem


def build_mean_spectra_from_overview_dir(config: OverviewBuildConfig) -> pd.DataFrame:
    input_dir = Path(config.input_dir)
    files = sorted(input_dir.glob(config.file_pattern))
    if not files:
        raise ValueError(f"在 {input_dir} 下没有找到匹配 {config.file_pattern!r} 的文件。")

    if config.labels is not None:
        labels = list(config.labels)
        if len(labels) != len(files):
            raise ValueError("labels 数量必须与 overview spectra 文件数量一致。")
    elif config.label_from_filename:
        labels = [infer_label_from_path(p) for p in files]
    else:
        labels = [f"group_{i + 1}" for i in range(len(files))]

    merged: pd.DataFrame | None = None
    for path, label in zip(files, labels):
        _, df = read_overview_spectrum(str(path), mz_col=config.mz_col, intensity_col=config.intensity_col)
        renamed = df.rename(columns={"intensity": label})
        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on="mz", how=config.align_how)

    assert merged is not None
    merged = merged.sort_values("mz").reset_index(drop=True)

    numeric_cols = [c for c in merged.columns if c != "mz"]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    out_path = Path(config.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return merged


def read_scils_featurelist(path: str) -> Tuple[List[str], pd.DataFrame]:
    comments, df = _read_scils_semicolon_with_comments(path)
    if "m/z" not in df.columns:
        raise ValueError("SCiLS feature list 中缺少 'm/z' 列。")

    df["m/z"] = pd.to_numeric(df["m/z"], errors="coerce")
    if "Interval Width (+/- Da)" in df.columns:
        df["Interval Width (+/- Da)"] = pd.to_numeric(df["Interval Width (+/- Da)"], errors="coerce")
    df = df.dropna(subset=["m/z"]).reset_index(drop=True)
    return comments, df


def nearest_feature(features: pd.DataFrame, target_mz: float) -> Tuple[float, float]:
    row = features.iloc[(features["m/z"] - target_mz).abs().argmin()]
    half_width = float(row["Interval Width (+/- Da)"]) if "Interval Width (+/- Da)" in row else np.nan
    return float(row["m/z"]), half_width
