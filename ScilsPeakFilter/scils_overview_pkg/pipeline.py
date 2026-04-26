from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from scils_peak_pkg import PeakWindowConfig, run_peak_window_pipeline

from .io import OverviewBuildConfig, build_mean_spectra_from_overview_dir, nearest_feature, read_scils_featurelist
from .plotting import plot_overview, plot_peak_window


@dataclass
class OverviewPipelineConfig:
    overview: OverviewBuildConfig
    peak_windows: PeakWindowConfig
    featurelist: str
    plot_overview_png: Optional[str] = None
    plot_targets: Optional[Sequence[float]] = None
    plot_all_features: bool = False
    peak_plot_dir: Optional[str] = None


def run_overview_pipeline(config: OverviewPipelineConfig) -> Dict[str, pd.DataFrame]:
    mean_df = build_mean_spectra_from_overview_dir(config.overview)

    if config.plot_overview_png:
        plot_overview(mean_df, config.plot_overview_png)

    if config.plot_targets or config.plot_all_features:
        _, features = read_scils_featurelist(config.featurelist)
        peak_dir = Path(config.peak_plot_dir) if config.peak_plot_dir else Path(Path(config.overview.out_csv).parent) / "peak_plots"
        peak_dir.mkdir(parents=True, exist_ok=True)

        if config.plot_all_features:
            plot_items = [
                (
                    float(row["m/z"]),
                    float(row["Interval Width (+/- Da)"]) if "Interval Width (+/- Da)" in row else None,
                )
                for _, row in features.iterrows()
            ]
        else:
            plot_items = [
                nearest_feature(features, float(target_mz))
                for target_mz in config.plot_targets or []
            ]

        for feature_mz, half_width in plot_items:
            plot_peak_window(
                mean_df=mean_df,
                center=feature_mz,
                half_width=half_width,
                search_ppm=config.peak_windows.search_ppm,
                out_path=str(peak_dir / f"peak_{feature_mz:.4f}.png"),
            )

    return run_peak_window_pipeline(config.peak_windows)
