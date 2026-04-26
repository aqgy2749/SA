#!/usr/bin/env python3
from pathlib import Path
import argparse

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


DEFAULT_MEAN_SPECTRA = Path(
    "/p2/zulab/jtian/data/SA/03_removeLow-qualityPeaks/ScilsPeakFilter/"
    "output_run_scils_pipeline/mean_spectra.csv"
)
DEFAULT_FEATURELIST = Path(
    "/p2/zulab/jtian/data/SA/03_removeLow-qualityPeaks/ScilsPeakFilter/2326featureList.csv"
)
DEFAULT_OUTDIR = Path("/p1/zulab_users/jtian/scils_peak_plots")
GROUPS = ["01", "02", "03", "04", "05"]
COLORS = ["#e84a5f", "#2ab07f", "#3973d6", "#8b5fbf", "#f59e0b"]


def read_featurelist(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path, sep=";", comment="#", encoding="utf-8-sig")
    return features.rename(columns={"m/z": "mz", "Interval Width (+/- Da)": "half_width"})


def nearest_feature(features: pd.DataFrame, target_mz: float) -> tuple[float, float]:
    row = features.iloc[(features["mz"] - target_mz).abs().argmin()]
    return float(row["mz"]), float(row["half_width"])


def plot_overview(mean_spectra: Path, features: pd.DataFrame, outdir: Path) -> Path:
    overview_parts = []
    for chunk in pd.read_csv(mean_spectra, chunksize=300_000):
        sub = chunk.iloc[::500].copy()
        sub["mean_all"] = sub[GROUPS].mean(axis=1)
        overview_parts.append(sub[["mz", "mean_all"]])
    overview = pd.concat(overview_parts, ignore_index=True)

    fig, ax = plt.subplots(figsize=(15, 5), dpi=180)
    ax.plot(overview["mz"], overview["mean_all"], color="#1f2937", lw=0.8)
    ax.vlines(
        features["mz"],
        ymin=0,
        ymax=max(overview["mean_all"].max() * 0.025, 1),
        color="#e879f9",
        alpha=0.15,
        lw=0.3,
    )
    ax.set_title("Mean spectra overview (5 groups averaged)", fontsize=13, weight="bold")
    ax.set_xlabel("m/z", fontsize=11, weight="bold")
    ax.set_ylabel("Mean absolute intensity", fontsize=11, weight="bold")
    ax.set_xlim(float(overview["mz"].min()), float(overview["mz"].max()))
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    output = outdir / "mean_spectra_overview.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def read_window(mean_spectra: Path, center: float, half_window: float) -> pd.DataFrame:
    rows = []
    for chunk in pd.read_csv(mean_spectra, chunksize=300_000):
        sub = chunk[(chunk["mz"] >= center - half_window) & (chunk["mz"] <= center + half_window)]
        if not sub.empty:
            rows.append(sub.copy())
        if chunk["mz"].iloc[0] > center + half_window:
            break
    if not rows:
        raise ValueError(f"No mean spectra rows found near m/z {center}")
    return pd.concat(rows, ignore_index=True)


def plot_peak(mean_spectra: Path, outdir: Path, center: float, half_width: float, search_ppm: float) -> tuple[Path, Path]:
    plot_half = max(half_width * 2.2, 0.004)
    win = read_window(mean_spectra, center, plot_half)
    search_half = center * search_ppm / 1e6

    fig, ax = plt.subplots(figsize=(16, 4.1), dpi=220)
    ax.axvspan(
        center - half_width,
        center + half_width,
        color="#e879f9",
        alpha=0.32,
        label=f"SCiLS interval +/- {half_width:.6f} Da",
    )
    ax.axvspan(
        center - search_half,
        center + search_half,
        color="#60a5fa",
        alpha=0.16,
        label=f"current search_ppm={search_ppm:g}",
    )

    for group, color in zip(GROUPS, COLORS):
        ax.plot(win["mz"], win[group], color=color, lw=1.15, alpha=0.95, label=group)
        imax = win[group].idxmax()
        ax.vlines(
            win.loc[imax, "mz"],
            ymin=win[group].min(),
            ymax=win.loc[imax, group],
            color=color,
            lw=0.8,
            alpha=0.55,
        )

    mean_y = win[GROUPS].mean(axis=1)
    peak_idx = mean_y.idxmax()
    peak_mz = float(win.loc[peak_idx, "mz"])
    ax.axvline(center, color="#2563eb", lw=1.0, alpha=0.9, label=f"feature m/z {center:.4f}")
    ax.axvline(peak_mz, color="#111827", lw=0.8, ls="--", alpha=0.7, label=f"mean peak {peak_mz:.6f}")

    ax.set_title(f"Mean spectra peak near m/z {center:.4f}", fontsize=13, weight="bold")
    ax.set_xlabel("m/z", fontsize=11, weight="bold")
    ax.set_ylabel("Absolute intensity", fontsize=11, weight="bold")
    ax.set_xlim(center - plot_half, center + plot_half)
    ax.set_ylim(bottom=max(0, float(win[GROUPS].min().min()) * 0.98))
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", ncol=4, frameon=True, fontsize=8)
    fig.tight_layout()

    image_output = outdir / f"mean_spectra_peak_{center:.4f}.png"
    fig.savefig(image_output, bbox_inches="tight")
    plt.close(fig)

    diagnostic_output = outdir / f"mean_spectra_peak_{center:.4f}_diagnostic.txt"
    with diagnostic_output.open("w") as f:
        f.write(f"source_mean_spectra\t{mean_spectra}\n")
        f.write(f"feature_mz\t{center:.10f}\n")
        f.write(f"scils_interval_half_width_da\t{half_width:.10f}\n")
        f.write(f"search_ppm\t{search_ppm:g}\n")
        f.write(f"search_half_width_da\t{search_half:.10f}\n")
        f.write(f"mean_peak_mz\t{peak_mz:.10f}\n")
        for group in GROUPS:
            imax = win[group].idxmax()
            f.write(
                f"{group}_peak_mz\t{win.loc[imax, 'mz']:.10f}"
                f"\t{group}_peak_intensity\t{win.loc[imax, group]:.6f}\n"
            )

    return image_output, diagnostic_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean_spectra.csv in a SCiLS-like style.")
    parser.add_argument("--mean-spectra", type=Path, default=DEFAULT_MEAN_SPECTRA)
    parser.add_argument("--featurelist", type=Path, default=DEFAULT_FEATURELIST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--target-mz", type=float, default=85.0294)
    parser.add_argument("--search-ppm", type=float, default=10.0)
    parser.add_argument("--skip-overview", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    features = read_featurelist(args.featurelist)
    center, half_width = nearest_feature(features, args.target_mz)

    outputs = []
    if not args.skip_overview:
        outputs.append(plot_overview(args.mean_spectra, features, args.outdir))
    outputs.extend(plot_peak(args.mean_spectra, args.outdir, center, half_width, args.search_ppm))

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
