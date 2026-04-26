#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter


MEAN_SPECTRA = Path(
    "/p2/zulab/jtian/data/SA/03_removeLow-qualityPeaks/ScilsPeakFilter/"
    "output_run_scils_pipeline/mean_spectra.csv"
)
FEATURELIST = Path(
    "/p2/zulab/jtian/data/SA/03_removeLow-qualityPeaks/ScilsPeakFilter/2326featureList.csv"
)
OUTDIR = Path("/p1/zulab_users/jtian/scils_peak_plots/mz_0_200")

MZ_MIN = 0.0
MZ_MAX = 200.0
SEARCH_PPM = 10.0
GROUPS = ["01", "02", "03", "04", "05"]
COLORS = ["#e84a5f", "#2ab07f", "#3973d6", "#8b5fbf", "#f59e0b"]


def read_features() -> pd.DataFrame:
    features = pd.read_csv(FEATURELIST, sep=";", comment="#", encoding="utf-8-sig")
    features = features.rename(columns={"m/z": "mz", "Interval Width (+/- Da)": "half_width"})
    features = features[(features["mz"] >= MZ_MIN) & (features["mz"] <= MZ_MAX)].copy()
    features = features.sort_values("mz").reset_index(drop=True)
    return features[["mz", "half_width"]]


def read_mean_spectra_window(features: pd.DataFrame) -> pd.DataFrame:
    margin = max(float(features["half_width"].max()) * 3.0, 0.01)
    lo = max(MZ_MIN, float(features["mz"].min()) - margin)
    hi = min(MZ_MAX, float(features["mz"].max()) + margin)

    rows = []
    for chunk in pd.read_csv(MEAN_SPECTRA, chunksize=300_000):
        sub = chunk[(chunk["mz"] >= lo) & (chunk["mz"] <= hi)]
        if not sub.empty:
            rows.append(sub.copy())
        if chunk["mz"].iloc[0] > hi:
            break

    if not rows:
        raise RuntimeError(f"No rows found in mean_spectra.csv for m/z {lo:.4f}-{hi:.4f}")
    return pd.concat(rows, ignore_index=True)


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_overview(spectra: pd.DataFrame, features: pd.DataFrame, outdir: Path) -> Path:
    overview = spectra.iloc[::20].copy()
    overview["mean_all"] = overview[GROUPS].mean(axis=1)

    fig, ax = plt.subplots(figsize=(18, 5), dpi=180)
    for group, color in zip(GROUPS, COLORS):
        ax.plot(overview["mz"], overview[group], color=color, lw=0.55, alpha=0.55, label=group)
    ax.plot(overview["mz"], overview["mean_all"], color="#111827", lw=0.9, alpha=0.9, label="mean")
    ax.vlines(
        features["mz"],
        ymin=0,
        ymax=max(float(overview["mean_all"].max()) * 0.035, 1.0),
        color="#e879f9",
        alpha=0.22,
        lw=0.35,
    )

    ax.set_title("Mean spectra, m/z 0-200", fontsize=13, weight="bold")
    ax.set_xlabel("m/z", fontsize=11, weight="bold")
    ax.set_ylabel("Absolute intensity", fontsize=11, weight="bold")
    ax.set_xlim(MZ_MIN, MZ_MAX)
    ax.legend(loc="upper right", ncol=6, frameon=True, fontsize=8)
    style_axis(ax)
    fig.tight_layout()

    output = outdir / "mean_spectra_mz_0_200_overview.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_peak_page(ax, spectra: pd.DataFrame, center: float, half_width: float) -> dict:
    plot_half = max(half_width * 2.2, 0.004)
    win = spectra[(spectra["mz"] >= center - plot_half) & (spectra["mz"] <= center + plot_half)]
    if win.empty:
        raise RuntimeError(f"No local rows found for m/z {center:.4f}")

    search_half = center * SEARCH_PPM / 1e6
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
        label=f"search_ppm={SEARCH_PPM:g}",
    )

    peak_info = {
        "feature_mz": center,
        "scils_interval_half_width_da": half_width,
        "search_half_width_da": search_half,
    }
    for group, color in zip(GROUPS, COLORS):
        ax.plot(win["mz"], win[group], color=color, lw=1.15, alpha=0.95, label=group)
        imax = win[group].idxmax()
        peak_info[f"{group}_peak_mz"] = float(win.loc[imax, "mz"])
        peak_info[f"{group}_peak_intensity"] = float(win.loc[imax, group])
        ax.vlines(
            win.loc[imax, "mz"],
            ymin=win[group].min(),
            ymax=win.loc[imax, group],
            color=color,
            lw=0.8,
            alpha=0.55,
        )

    mean_y = win[GROUPS].mean(axis=1)
    mean_peak_idx = mean_y.idxmax()
    peak_info["mean_peak_mz"] = float(win.loc[mean_peak_idx, "mz"])
    peak_info["mean_peak_intensity"] = float(mean_y.loc[mean_peak_idx])

    ax.axvline(center, color="#2563eb", lw=1.0, alpha=0.9, label=f"feature {center:.4f}")
    ax.axvline(
        peak_info["mean_peak_mz"],
        color="#111827",
        lw=0.8,
        ls="--",
        alpha=0.7,
        label=f"mean peak {peak_info['mean_peak_mz']:.6f}",
    )
    ax.set_title(f"Mean spectra peak near m/z {center:.4f}", fontsize=13, weight="bold")
    ax.set_xlabel("m/z", fontsize=11, weight="bold")
    ax.set_ylabel("Absolute intensity", fontsize=11, weight="bold")
    ax.set_xlim(center - plot_half, center + plot_half)
    ax.set_ylim(bottom=max(0, float(win[GROUPS].min().min()) * 0.98))
    ax.legend(loc="upper right", ncol=4, frameon=True, fontsize=7.5)
    style_axis(ax)
    return peak_info


def plot_peak_pdf(spectra: pd.DataFrame, features: pd.DataFrame, outdir: Path) -> tuple[Path, Path]:
    pdf_path = outdir / "mean_spectra_mz_0_200_scils_like_peaks.pdf"
    csv_path = outdir / "mean_spectra_mz_0_200_peak_apex_summary.csv"
    records = []

    with PdfPages(pdf_path) as pdf:
        for _, row in features.iterrows():
            fig, ax = plt.subplots(figsize=(16, 4.1), dpi=160)
            records.append(plot_peak_page(ax, spectra, float(row["mz"]), float(row["half_width"])))
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pd.DataFrame(records).to_csv(csv_path, index=False)
    return pdf_path, csv_path


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    features = read_features()
    spectra = read_mean_spectra_window(features)

    overview_png = plot_overview(spectra, features, OUTDIR)
    peaks_pdf, summary_csv = plot_peak_pdf(spectra, features, OUTDIR)

    print(f"features_0_200: {len(features)}")
    print(overview_png)
    print(peaks_pdf)
    print(summary_csv)


if __name__ == "__main__":
    main()
