from pathlib import Path

from scils_peak_pkg import MeanSpectraConfig, PeakWindowConfig, PipelineConfig, run_full_pipeline


# =========================================================
# 只改这里：路径和参数区
# =========================================================
PROJECT_DIR = Path("/p2/zulab/jtian/data/SA/03_removeLow-qualityPeaks/ScilsPeakFilter/")
ROI_DIR =  Path("/p2/zulab/jtian/data/SA/03_removeLow-qualityPeaks/imzML/")
OUT_DIR = PROJECT_DIR / "output_run_scils_pipeline"

INPUTS = [
    str(ROI_DIR / "01.imzML"),
    str(ROI_DIR / "02.imzML"),
    str(ROI_DIR / "03.imzML"),
    str(ROI_DIR / "04.imzML"),
    str(ROI_DIR / "05.imzML"),
]

LABELS = ["01", "02", "03", "04", "05"]
FEATURELIST = str(PROJECT_DIR / "2326featureList.csv")
# FEATURELIST = str(PROJECT_DIR / "only_in_ctrl3_460.csv")
MEAN_SPECTRA_CSV = str(OUT_DIR / "mean_spectra.csv")

mean_cfg = MeanSpectraConfig(
    inputs=INPUTS,
    labels=LABELS,
    out=MEAN_SPECTRA_CSV,
    force_processed=True,
    mz_step=0.0001,
    processed_projection="interp",   # 你当前想模拟 SCiLS-like 公共轴平均，用 interp
)

peak_cfg = PeakWindowConfig(
    featurelist=FEATURELIST,
    mean_spectra=MEAN_SPECTRA_CSV,
    outdir=str(OUT_DIR),
    do_smooth=False,
    do_baseline=False,
    smooth_window=11,
    smooth_polyorder=3,
    baseline_window=101,
    search_ppm=10.0,
    prominence_frac=0.05,
    snr_threshold=3.0,
    fwhm_min_ppm=3.0,
    fwhm_max_ppm=200.0,
    valley_ratio_max=0.50,
    secondary_peak_ratio_max=0.70,
    base_rel_height=0.10,
    require_all_groups=True,
)

pipeline_cfg = PipelineConfig(
    mean_spectra=mean_cfg,
    peak_windows=peak_cfg,
)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_full_pipeline(pipeline_cfg)

    print("Done.")
    print(f"mean_spectra.csv -> {MEAN_SPECTRA_CSV}")
    print(f"group_peak_details.csv -> {OUT_DIR / 'group_peak_details.csv'}")
    print(f"seed_summary.csv -> {OUT_DIR / 'seed_summary.csv'}")
    print(f"scils_import_minimal.csv -> {OUT_DIR / 'scils_import_minimal.csv'}")
    print(f"scils_featurelist_patched.csv -> {OUT_DIR / 'scils_featurelist_patched.csv'}")
