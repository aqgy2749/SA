from __future__ import annotations

from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


@dataclass
class GroupPeakResult:
    group: str
    found: bool
    independent: bool
    seed_mz: float
    peak_mz: Optional[float] = None
    peak_intensity: Optional[float] = None
    snr: Optional[float] = None
    left_half_mz: Optional[float] = None
    right_half_mz: Optional[float] = None
    fwhm_da: Optional[float] = None
    fwhm_ppm: Optional[float] = None
    left_base_mz: Optional[float] = None
    right_base_mz: Optional[float] = None
    left_valley_ratio: Optional[float] = None
    right_valley_ratio: Optional[float] = None
    reason: str = ""


@dataclass
class PeakWindowConfig:
    featurelist: str
    mean_spectra: str
    outdir: str = "scils_peak_window_out"
    do_smooth: bool = True
    do_baseline: bool = False
    smooth_window: int = 11
    smooth_polyorder: int = 3
    baseline_window: int = 101
    search_ppm: float = 30.0
    prominence_frac: float = 0.05
    snr_threshold: float = 3.0
    fwhm_min_ppm: float = 3.0
    fwhm_max_ppm: float = 200.0
    valley_ratio_max: float = 0.60
    secondary_peak_ratio_max: float = 0.70
    base_rel_height: float = 0.10
    require_all_groups: bool = True
    min_independent_groups: int = 5


def read_scils_featurelist(path: str) -> Tuple[List[str], pd.DataFrame]:
    comments: List[str] = []
    data_lines: List[str] = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line)
            elif line.strip():
                data_lines.append(line)

    if not data_lines:
        raise ValueError("No tabular data found in SCiLS feature list export.")

    df = pd.read_csv(StringIO("".join(data_lines)), sep=";")
    return comments, df


def read_mean_spectra(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "mz" not in df.columns:
        raise ValueError("mean spectrum CSV 必须包含列名 'mz'。")
    df = df.sort_values("mz").reset_index(drop=True)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["mz"].isna().any():
        raise ValueError("mz 列存在无法解析的值。")

    group_cols = [c for c in df.columns if c != "mz"]
    if len(group_cols) < 2:
        raise ValueError("至少需要 mz + 2 组 intensity 列。")

    return df


def moving_min_baseline(y: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(y)
    baseline = (
        s.rolling(window=window, center=True, min_periods=1).min()
         .rolling(window=window, center=True, min_periods=1).max()
         .to_numpy()
    )
    return baseline


def robust_sigma(y: np.ndarray) -> float:
    mad = np.median(np.abs(y - np.median(y)))
    sigma = 1.4826 * mad
    return float(max(sigma, 1e-12))


def preprocess_signal(
    y: np.ndarray,
    do_smooth: bool = True,
    do_baseline: bool = False,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    baseline_window: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if do_smooth and len(y) >= smooth_window and smooth_window >= 5:
        if smooth_window % 2 == 0:
            smooth_window += 1
        poly = min(smooth_polyorder, smooth_window - 2)
        y_smooth = savgol_filter(y, smooth_window, poly)
    else:
        y_smooth = y.copy()

    if do_baseline:
        if baseline_window % 2 == 0:
            baseline_window += 1
        baseline = moving_min_baseline(y_smooth, baseline_window)
        y_corr = y_smooth - baseline
        y_corr[y_corr < 0] = 0.0
    else:
        y_corr = y_smooth.copy()

    return y_smooth, y_corr


def ppm_to_da(mz: float, ppm: float) -> float:
    return mz * ppm / 1e6


def interp_x_at_y(x1, y1, x2, y2, y_target):
    if y2 == y1:
        return (x1 + x2) / 2
    t = (y_target - y1) / (y2 - y1)
    t = np.clip(t, 0.0, 1.0)
    return x1 + t * (x2 - x1)


def compute_absolute_fwhm(mz: np.ndarray, y: np.ndarray, peak_idx: int):
    peak_h = float(y[peak_idx])
    if peak_h <= 0:
        return None

    half_h = peak_h / 2.0

    i = peak_idx
    while i > 0 and y[i] >= half_h:
        i -= 1
    if i == peak_idx:
        return None
    left_mz = interp_x_at_y(mz[i], y[i], mz[i + 1], y[i + 1], half_h)

    j = peak_idx
    while j < len(y) - 1 and y[j] >= half_h:
        j += 1
    if j == peak_idx or j >= len(y):
        return None
    right_mz = interp_x_at_y(mz[j - 1], y[j - 1], mz[j], y[j], half_h)

    if right_mz <= left_mz:
        return None

    center = float(mz[peak_idx])
    fwhm_da = right_mz - left_mz
    fwhm_ppm = fwhm_da / center * 1e6
    return left_mz, right_mz, fwhm_da, fwhm_ppm


def compute_base_window(
    mz: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    rel_height: float = 0.10,
) -> Tuple[float, float]:
    peak_h = float(y[peak_idx])
    thr = peak_h * rel_height

    i = peak_idx
    while i > 0 and y[i] > thr:
        i -= 1
    if i < len(y) - 1:
        left_base = interp_x_at_y(mz[i], y[i], mz[i + 1], y[i + 1], thr)
    else:
        left_base = mz[i]

    j = peak_idx
    while j < len(y) - 1 and y[j] > thr:
        j += 1
    if j > 0:
        right_base = interp_x_at_y(mz[j - 1], y[j - 1], mz[j], y[j], thr)
    else:
        right_base = mz[j]

    return float(left_base), float(right_base)


def valley_ratio_near_peak(y: np.ndarray, main_idx: int, other_peak_indices: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    peak_h = float(y[main_idx])
    if peak_h <= 0:
        return None, None

    left_peaks = other_peak_indices[other_peak_indices < main_idx]
    right_peaks = other_peak_indices[other_peak_indices > main_idx]

    left_ratio = None
    right_ratio = None

    if len(left_peaks) > 0:
        lp = left_peaks[-1]
        valley = np.min(y[lp:main_idx + 1])
        left_ratio = float(valley / peak_h)

    if len(right_peaks) > 0:
        rp = right_peaks[0]
        valley = np.min(y[main_idx:rp + 1])
        right_ratio = float(valley / peak_h)

    return left_ratio, right_ratio


def analyze_group_peak(
    mz_axis: np.ndarray,
    intensity: np.ndarray,
    group_name: str,
    seed_mz: float,
    search_ppm: float = 30.0,
    prominence_frac: float = 0.05,
    snr_threshold: float = 3.0,
    fwhm_min_ppm: float = 3.0,
    fwhm_max_ppm: float = 200.0,
    valley_ratio_max: float = 0.60,
    secondary_peak_ratio_max: float = 0.70,
    base_rel_height: float = 0.10,
) -> GroupPeakResult:
    tol_da = ppm_to_da(seed_mz, search_ppm)
    mask = (mz_axis >= seed_mz - tol_da) & (mz_axis <= seed_mz + tol_da)

    if mask.sum() < 7:
        return GroupPeakResult(group=group_name, found=False, independent=False, seed_mz=seed_mz, reason="local window too small")

    mz = mz_axis[mask]
    y = intensity[mask]

    sigma = robust_sigma(y)
    prominence = max(float(np.max(y)) * prominence_frac, sigma * 3)

    peaks, _ = find_peaks(y, prominence=prominence)
    if len(peaks) == 0:
        return GroupPeakResult(group=group_name, found=False, independent=False, seed_mz=seed_mz, reason="no local peak found")

    distances = np.abs(mz[peaks] - seed_mz)
    order = np.lexsort((-y[peaks], distances))
    main_idx_local = peaks[order[0]]

    peak_mz = float(mz[main_idx_local])
    peak_h = float(y[main_idx_local])
    snr = peak_h / sigma if sigma > 0 else np.inf

    if snr < snr_threshold:
        return GroupPeakResult(
            group=group_name, found=True, independent=False, seed_mz=seed_mz,
            peak_mz=peak_mz, peak_intensity=peak_h, snr=snr, reason="low SNR"
        )

    fwhm = compute_absolute_fwhm(mz, y, int(main_idx_local))
    if fwhm is None:
        return GroupPeakResult(
            group=group_name, found=True, independent=False, seed_mz=seed_mz,
            peak_mz=peak_mz, peak_intensity=peak_h, snr=snr, reason="cannot compute FWHM"
        )

    left_half_mz, right_half_mz, fwhm_da, fwhm_ppm = fwhm
    if fwhm_ppm < fwhm_min_ppm or fwhm_ppm > fwhm_max_ppm:
        return GroupPeakResult(
            group=group_name, found=True, independent=False, seed_mz=seed_mz,
            peak_mz=peak_mz, peak_intensity=peak_h, snr=snr,
            left_half_mz=left_half_mz, right_half_mz=right_half_mz,
            fwhm_da=fwhm_da, fwhm_ppm=fwhm_ppm,
            reason="FWHM out of range"
        )

    left_base_mz, right_base_mz = compute_base_window(mz, y, int(main_idx_local), rel_height=base_rel_height)
    other_peaks = peaks[peaks != main_idx_local]
    left_ratio, right_ratio = valley_ratio_near_peak(y, int(main_idx_local), other_peaks)

    independent = True
    reason_parts: List[str] = []

    for p in other_peaks:
        if y[p] >= secondary_peak_ratio_max * peak_h:
            independent = False
            reason_parts.append("competing secondary peak")
            break

    if left_ratio is not None and left_ratio > valley_ratio_max:
        independent = False
        reason_parts.append("left valley too shallow")

    if right_ratio is not None and right_ratio > valley_ratio_max:
        independent = False
        reason_parts.append("right valley too shallow")

    reason = "; ".join(reason_parts) if reason_parts else "ok"

    return GroupPeakResult(
        group=group_name,
        found=True,
        independent=independent,
        seed_mz=seed_mz,
        peak_mz=peak_mz,
        peak_intensity=peak_h,
        snr=snr,
        left_half_mz=left_half_mz,
        right_half_mz=right_half_mz,
        fwhm_da=fwhm_da,
        fwhm_ppm=fwhm_ppm,
        left_base_mz=left_base_mz,
        right_base_mz=right_base_mz,
        left_valley_ratio=left_ratio,
        right_valley_ratio=right_ratio,
        reason=reason,
    )


def analyze_seed_across_groups(
    seed_mz: float,
    mean_df: pd.DataFrame,
    group_cols: List[str],
    search_ppm: float,
    prominence_frac: float,
    snr_threshold: float,
    fwhm_min_ppm: float,
    fwhm_max_ppm: float,
    valley_ratio_max: float,
    secondary_peak_ratio_max: float,
    base_rel_height: float,
    require_all_groups: bool,
    min_independent_groups: int,
) -> Tuple[pd.DataFrame, Dict]:
    mz_axis = mean_df["mz"].to_numpy()
    group_results = []

    for g in group_cols:
        intensity = mean_df[g].to_numpy()
        result = analyze_group_peak(
            mz_axis=mz_axis,
            intensity=intensity,
            group_name=g,
            seed_mz=seed_mz,
            search_ppm=search_ppm,
            prominence_frac=prominence_frac,
            snr_threshold=snr_threshold,
            fwhm_min_ppm=fwhm_min_ppm,
            fwhm_max_ppm=fwhm_max_ppm,
            valley_ratio_max=valley_ratio_max,
            secondary_peak_ratio_max=secondary_peak_ratio_max,
            base_rel_height=base_rel_height,
        )
        group_results.append(asdict(result))

    gdf = pd.DataFrame(group_results)

    n_found = int(gdf["found"].sum())
    n_independent = int(gdf["independent"].sum())
    all_found = bool(gdf["found"].all())
    all_independent = bool(gdf["independent"].all())

    if require_all_groups:
        passed = all_found and all_independent
    else:
        passed = n_independent >= min_independent_groups

    summary = {
        "seed_mz": seed_mz,
        "all_found": all_found,
        "all_independent": all_independent,
        "passed": passed,
        "n_found": n_found,
        "n_independent": n_independent,
        "consensus_peak_mz": np.nan,
        "shared_left_base_mz": np.nan,
        "shared_right_base_mz": np.nan,
        "shared_window_width_da": np.nan,
        "scils_center_mz": np.nan,
        "scils_interval_half_width_da": np.nan,
        "median_fwhm_ppm": np.nan,
    }

    good = gdf[gdf["independent"] == True].copy()
    if len(good) > 0:
        summary["median_fwhm_ppm"] = float(good["fwhm_ppm"].median())

    if passed and len(good) > 0:
        consensus_peak_mz = float(good["peak_mz"].median())
        left_mz = float(good["left_base_mz"].min())
        right_mz = float(good["right_base_mz"].max())
        width_da = right_mz - left_mz
        half_width_for_scils = max(consensus_peak_mz - left_mz, right_mz - consensus_peak_mz)

        summary.update({
            "consensus_peak_mz": consensus_peak_mz,
            "shared_left_base_mz": left_mz,
            "shared_right_base_mz": right_mz,
            "shared_window_width_da": width_da,
            "scils_center_mz": consensus_peak_mz,
            "scils_interval_half_width_da": float(half_width_for_scils),
        })

    return gdf, summary


def build_scils_minimal_import(summary_df: pd.DataFrame) -> pd.DataFrame:
    passed = summary_df[summary_df["passed"] == True].copy()
    out = pd.DataFrame({
        "m/z": passed["scils_center_mz"],
        "Interval Width (+/- Da)": passed["scils_interval_half_width_da"],
        "Color": "#ff00ff",
        "Name": [f"auto_pk_{i + 1}" for i in range(len(passed))],
    })
    return out


def write_scils_semicolon_csv(df: pd.DataFrame, out_path: str, comments: Optional[List[str]] = None):
    with open(out_path, "w", encoding="utf-8-sig") as f:
        if comments is not None:
            for line in comments:
                f.write(line if line.endswith("\n") else line + "\n")
        df.to_csv(f, sep=";", index=False)


def run_peak_window_pipeline(config: PeakWindowConfig) -> Dict[str, pd.DataFrame]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comments, feature_df = read_scils_featurelist(config.featurelist)
    mean_df = read_mean_spectra(config.mean_spectra)

    group_cols = [c for c in mean_df.columns if c != "mz"]
    if config.require_all_groups:
        config.min_independent_groups = len(group_cols)

    processed = pd.DataFrame({"mz": mean_df["mz"].to_numpy()})
    for g in group_cols:
        y = mean_df[g].to_numpy()
        _, y_corr = preprocess_signal(
            y=y,
            do_smooth=config.do_smooth,
            do_baseline=config.do_baseline,
            smooth_window=config.smooth_window,
            smooth_polyorder=config.smooth_polyorder,
            baseline_window=config.baseline_window,
        )
        processed[g] = y_corr

    seed_list = feature_df["m/z"].astype(float).tolist()
    all_group_rows = []
    all_summary_rows = []

    for seed_mz in seed_list:
        gdf, summary = analyze_seed_across_groups(
            seed_mz=seed_mz,
            mean_df=processed,
            group_cols=group_cols,
            search_ppm=config.search_ppm,
            prominence_frac=config.prominence_frac,
            snr_threshold=config.snr_threshold,
            fwhm_min_ppm=config.fwhm_min_ppm,
            fwhm_max_ppm=config.fwhm_max_ppm,
            valley_ratio_max=config.valley_ratio_max,
            secondary_peak_ratio_max=config.secondary_peak_ratio_max,
            base_rel_height=config.base_rel_height,
            require_all_groups=config.require_all_groups,
            min_independent_groups=config.min_independent_groups,
        )
        all_group_rows.append(gdf)
        all_summary_rows.append(summary)

    group_detail_df = pd.concat(all_group_rows, ignore_index=True)
    summary_df = pd.DataFrame(all_summary_rows)

    group_detail_df.to_csv(outdir / "group_peak_details.csv", index=False)
    summary_df.to_csv(outdir / "seed_summary.csv", index=False)

    scils_min_df = build_scils_minimal_import(summary_df)
    write_scils_semicolon_csv(scils_min_df, str(outdir / "scils_import_minimal.csv"))

    passed = summary_df[summary_df["passed"] == True].copy()
    passed_map = dict(zip(passed["seed_mz"], zip(passed["scils_center_mz"], passed["scils_interval_half_width_da"])))

    patched = feature_df.copy()
    keep_mask = patched["m/z"].astype(float).isin(passed["seed_mz"].astype(float))
    patched = patched.loc[keep_mask].copy()
    patched["Name"] = [f"auto_pk_{i + 1}" for i in range(len(patched))]

    for idx in patched.index:
        old_mz = float(patched.at[idx, "m/z"])
        new_center, new_half_width = passed_map[old_mz]
        patched.at[idx, "m/z"] = new_center
        patched.at[idx, "Interval Width (+/- Da)"] = new_half_width

    write_scils_semicolon_csv(patched, str(outdir / "scils_featurelist_patched.csv"), comments=comments)

    return {
        "processed_mean_spectra": processed,
        "group_peak_details": group_detail_df,
        "seed_summary": summary_df,
        "scils_import_minimal": scils_min_df,
        "scils_featurelist_patched": patched,
    }
