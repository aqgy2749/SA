from pathlib import Path

import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser


# ===== 参数区：换组/路径主要改这里 =====
PROJECT_DIR = Path("/p2/zulab/jtian/data/SA")
IMZML_DIR = PROJECT_DIR / "03_removeLow-qualityPeaks" / "imzML"
FEATURE_BASE_DIR = PROJECT_DIR / "03_removeLow-qualityPeaks" / "ScilsPeakFilter"
OUT_BASE_DIR = PROJECT_DIR / "06_calculateConcentration" / "MetaboliteIntensity" / "from_imzML_RMS_scilsPeakFilter"
MANUAL_BASE_DIR = PROJECT_DIR / "06_calculateConcentration" / "MetaboliteIntensity" / "scilsPeakFilter"

GROUPS = ["ctrl1", "ctrl2","ctrl3", "dn1", "dn2", "dn3"]
TIME_TO_IMZML = {
    "0": "01.imzML",
    "15": "02.imzML",
    "30": "03.imzML",
    "45": "04.imzML",
    "60": "05.imzML",
}

FEATURE_FILE_NAME = "scils_import_minimal.csv"
INTENSITY_MODE = "sum"
RMS_NORMALIZE = True
COMPARE_WITH_MANUAL = True
COMPARE_RTOL = 1e-5
COMPARE_ATOL = 1e-8


compare_rows = []

for group in GROUPS:
    print(f"\n===== Processing {group} =====")

    feature_csv = FEATURE_BASE_DIR / f"output_run_overview_pipeline_{group}" / FEATURE_FILE_NAME
    out_dir = OUT_BASE_DIR / group
    out_dir.mkdir(parents=True, exist_ok=True)
    

    features = pd.read_csv(feature_csv, sep=";", encoding="utf-8-sig")
    target_mz = features["m/z"].astype(float).to_numpy()
    half_width = features["Interval Width (+/- Da)"].astype(float).to_numpy()
    left_mz = target_mz - half_width
    right_mz = target_mz + half_width

    for tp, imzml_name in TIME_TO_IMZML.items():
        imzml_path = IMZML_DIR / imzml_name
        out_csv = out_dir / f"{tp}.csv"
        print(f"\n{group} time {tp}: {imzml_path}")

        parser = ImzMLParser(str(imzml_path))
        n_pixels = len(parser.coordinates)
        result = np.zeros((len(target_mz), n_pixels), dtype=np.float64)

        for pixel_i in range(n_pixels):
            mzs, intensities = parser.getspectrum(pixel_i)
            mzs = np.asarray(mzs, dtype=np.float64)########
            intensities = np.asarray(intensities, dtype=np.float64)

            ok = np.isfinite(mzs) & np.isfinite(intensities)
            mzs = mzs[ok]
            intensities = intensities[ok]

            if len(mzs) == 0:
                continue

            order = np.argsort(mzs)
            mzs = mzs[order]
            intensities = intensities[order]

            if RMS_NORMALIZE:
                rms = np.sqrt(np.mean(intensities ** 2))
                if np.isfinite(rms) and rms > 0:
                    intensities = intensities / rms
                else:
                    intensities = np.zeros_like(intensities)

            left_idx = np.searchsorted(mzs, left_mz, side="left")
            right_idx = np.searchsorted(mzs, right_mz, side="right")

            if INTENSITY_MODE == "sum":
                csum = np.concatenate(([0.0], np.cumsum(intensities)))
                result[:, pixel_i] = csum[right_idx] - csum[left_idx]
            elif INTENSITY_MODE == "max":
                pixel_values = np.zeros(len(target_mz), dtype=np.float64)
                for feature_i, (lo, hi) in enumerate(zip(left_idx, right_idx)):
                    if hi > lo:
                        pixel_values[feature_i] = np.max(intensities[lo:hi])
                result[:, pixel_i] = pixel_values
            else:
                raise ValueError("INTENSITY_MODE 只能是 'sum' 或 'max'。")

            if (pixel_i + 1) % 10000 == 0 or pixel_i + 1 == n_pixels:
                print(f"  pixels: {pixel_i + 1}/{n_pixels}")

        out_df = pd.DataFrame(
            result,
            index=target_mz,
            columns=[f"Spot {i + 1}" for i in range(n_pixels)],
        )
        out_df.index.name = "m/z"
        out_df.to_csv(out_csv, sep=";")
        print(f"Wrote: {out_csv} shape={out_df.shape}")

        if COMPARE_WITH_MANUAL:
            manual_csv = MANUAL_BASE_DIR / group / f"{tp}.csv"
            if not manual_csv.exists():
                print(f"Manual file not found, skip compare: {manual_csv}")
                compare_rows.append({
                    "group": group,
                    "time": tp,
                    "manual_csv": str(manual_csv),
                    "generated_csv": str(out_csv),
                    "manual_exists": False,
                    "same_shape": np.nan,
                    "same_index": np.nan,
                    "same_columns": np.nan,
                    "allclose": np.nan,
                    "max_abs_diff": np.nan,
                    "mean_abs_diff": np.nan,
                    "max_rel_diff": np.nan,
                })
                continue

            manual_df = pd.read_csv(manual_csv, sep=";", encoding="utf-8-sig", index_col=0)
            generated_df = pd.read_csv(out_csv, sep=";", encoding="utf-8-sig", index_col=0)

            manual_index = manual_df.index.astype(float).to_numpy()
            generated_index = generated_df.index.astype(float).to_numpy()

            same_shape = manual_df.shape == generated_df.shape
            same_index = (
                len(manual_index) == len(generated_index)
                and np.allclose(manual_index, generated_index, rtol=COMPARE_RTOL, atol=COMPARE_ATOL)
            )
            same_columns = list(manual_df.columns) == list(generated_df.columns)

            if same_shape:
                manual_values = manual_df.to_numpy(dtype=np.float64)
                generated_values = generated_df.to_numpy(dtype=np.float64)
                diff = generated_values - manual_values
                abs_diff = np.abs(diff)
                denom = np.maximum(np.abs(manual_values), COMPARE_ATOL)
                rel_diff = abs_diff / denom
                allclose = bool(np.allclose(generated_values, manual_values, rtol=COMPARE_RTOL, atol=COMPARE_ATOL))
                max_abs_diff = float(np.nanmax(abs_diff))
                mean_abs_diff = float(np.nanmean(abs_diff))
                max_rel_diff = float(np.nanmax(rel_diff))
            else:
                allclose = False
                max_abs_diff = np.nan
                mean_abs_diff = np.nan
                max_rel_diff = np.nan

            print(
                "Compare with manual:",
                f"same_shape={same_shape}",
                f"same_index={same_index}",
                f"same_columns={same_columns}",
                f"allclose={allclose}",
                f"max_abs_diff={max_abs_diff}",
            )

            compare_rows.append({
                "group": group,
                "time": tp,
                "manual_csv": str(manual_csv),
                "generated_csv": str(out_csv),
                "manual_exists": True,
                "same_shape": same_shape,
                "same_index": same_index,
                "same_columns": same_columns,
                "allclose": allclose,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "max_rel_diff": max_rel_diff,
            })


compare_df = pd.DataFrame(compare_rows)
compare_out = OUT_BASE_DIR / "compare_with_manual_summary.csv"
if not OUT_BASE_DIR.exists():
    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
compare_df.to_csv(compare_out, index=False)
print(f"\nCompare summary -> {compare_out}")
