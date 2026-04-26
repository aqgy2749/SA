from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter


DEFAULT_COLORS = ["#e84a5f", "#2ab07f", "#3973d6", "#8b5fbf", "#f59e0b", "#f59e0b", "#14b8a6"]


def _style_axis(ax) -> None:
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _group_cols(mean_df: pd.DataFrame) -> list[str]:
    cols = [c for c in mean_df.columns if c != "mz"]
    if not cols:
        raise ValueError("mean spectra 表至少需要 1 列 intensity。")
    return cols


def plot_overview(mean_df: pd.DataFrame, out_path: str, stride: int = 100) -> Path:
    group_cols = _group_cols(mean_df)
    view = mean_df.iloc[::max(1, stride)].copy()
    view["mean_all"] = view[group_cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=(16, 4.8), dpi=180)
    for idx, group in enumerate(group_cols):
        ax.plot(view["mz"], view[group], color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)], lw=0.55, alpha=0.5)
    ax.plot(view["mz"], view["mean_all"], color="#111827", lw=0.9, alpha=0.9)
    ax.set_title("SCiLS overview spectra", fontsize=13, weight="bold")
    ax.set_xlabel("m/z", fontsize=11, weight="bold")
    ax.set_ylabel("Absolute intensity", fontsize=11, weight="bold")
    ax.set_xlim(float(view["mz"].min()), float(view["mz"].max()))
    _style_axis(ax)
    fig.tight_layout()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_peak_window(
    mean_df: pd.DataFrame,
    center: float,
    out_path: str,
    half_width: Optional[float] = None,
    search_ppm: Optional[float] = None,
    plot_half: Optional[float] = None,
) -> Path:
    group_cols = _group_cols(mean_df)
    if plot_half is None:
        base_half = half_width if half_width is not None and half_width == half_width else 0.0015
        plot_half = max(base_half * 2.2, 0.004)

    win = mean_df[(mean_df["mz"] >= center - plot_half) & (mean_df["mz"] <= center + plot_half)].copy()
    if win.empty:
        raise ValueError(f"在 m/z {center:.6f} 附近没有找到数据。")

    fig, ax = plt.subplots(figsize=(16, 4.1), dpi=220)

    if half_width is not None and half_width == half_width:
        ax.axvspan(
            center - half_width,
            center + half_width,
            color="#e879f9",
            alpha=0.32,
            label=f"SCiLS interval +/- {half_width:.6f} Da",
        )
    if search_ppm is not None:
        search_half = center * search_ppm / 1e6
        ax.axvspan(
            center - search_half,
            center + search_half,
            color="#60a5fa",
            alpha=0.16,
            label=f"search_ppm={search_ppm:g}",
        )

    for idx, group in enumerate(group_cols):
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        ax.plot(win["mz"], win[group], color=color, lw=1.1, alpha=0.95, label=group)
        peak_idx = win[group].idxmax()
        ax.vlines(
            float(win.loc[peak_idx, "mz"]),
            ymin=float(win[group].min()),
            ymax=float(win.loc[peak_idx, group]),
            color=color,
            lw=0.8,
            alpha=0.55,
        )

    mean_y = win[group_cols].mean(axis=1)
    peak_idx = mean_y.idxmax()
    peak_mz = float(win.loc[peak_idx, "mz"])
    ax.axvline(center, color="#2563eb", lw=1.0, alpha=0.9, label=f"feature {center:.4f}")
    ax.axvline(peak_mz, color="#111827", lw=0.8, ls="--", alpha=0.7, label=f"mean peak {peak_mz:.6f}")

    ax.set_title(f"Mean spectra peak near m/z {center:.4f}", fontsize=13, weight="bold")
    ax.set_xlabel("m/z", fontsize=11, weight="bold")
    ax.set_ylabel("Absolute intensity", fontsize=11, weight="bold")
    ax.set_xlim(center - plot_half, center + plot_half)
    ax.set_ylim(bottom=max(0, float(win[group_cols].min().min()) * 0.98))
    ax.legend(loc="upper right", ncol=4, frameon=True, fontsize=8)
    _style_axis(ax)
    fig.tight_layout()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output
