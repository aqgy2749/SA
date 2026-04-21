from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .independent_windows import PeakWindowConfig, run_peak_window_pipeline
from .mean_spectra import MeanSpectraConfig, build_mean_spectra_csv


@dataclass
class PipelineConfig:
    mean_spectra: MeanSpectraConfig
    peak_windows: PeakWindowConfig


def run_full_pipeline(config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    build_mean_spectra_csv(config.mean_spectra)
    return run_peak_window_pipeline(config.peak_windows)
