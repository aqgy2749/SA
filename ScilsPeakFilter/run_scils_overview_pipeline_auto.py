from pathlib import Path

from scils_overview_pkg import OverviewBuildConfig, OverviewPipelineConfig, run_overview_pipeline
from scils_peak_pkg import PeakWindowConfig


PROJECT_DIR = Path("/p2/zulab/jtian/data/SA/03_removeLow-qualityPeaks/ScilsPeakFilter")
GROUP = "dn3"
OVERVIEW_DIR = PROJECT_DIR / "input" / "meanSpectra" / GROUP
OUT_DIR = PROJECT_DIR / f"output_run_overview_pipeline_auto_{GROUP}"
FEATURELIST = PROJECT_DIR / "2326featureList.csv"
MEAN_SPECTRA_CSV = OUT_DIR / "mean_spectra_from_overview.csv"


overview_cfg = OverviewBuildConfig(
    input_dir=str(OVERVIEW_DIR),
    out_csv=str(MEAN_SPECTRA_CSV),
    file_pattern="*-OverviewSpectra.csv",
    align_how="inner",
)

peak_cfg = PeakWindowConfig(
    featurelist=str(FEATURELIST),
    mean_spectra=str(MEAN_SPECTRA_CSV),
    outdir=str(OUT_DIR),
    seed_source="auto",
    do_smooth=False,
    do_baseline=False,
    smooth_window=11,
    smooth_polyorder=3,
    baseline_window=101,
    search_ppm=30.0,
    candidate_ppm=30.0,
    prominence_frac=0.05,
    snr_threshold=3.0,
    fwhm_min_ppm=3.0,
    fwhm_max_ppm=500.0,  # 200.0
    valley_ratio_max=0.50,
    secondary_peak_ratio_max=0.70,
    base_rel_height=0.10,
    require_all_groups=False,
    min_independent_groups=4,
    min_candidate_groups=4,
)

pipeline_cfg = OverviewPipelineConfig(
    overview=overview_cfg,
    peak_windows=peak_cfg,
    featurelist=str(FEATURELIST),
    plot_overview_png=str(OUT_DIR / "overview_from_scils_export.png"),
    plot_all_features=True,
    peak_plot_dir=str(OUT_DIR / "peak_plots"),
)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_overview_pipeline(pipeline_cfg)

    print("Done.")
    print(f"mean_spectra_from_overview.csv -> {MEAN_SPECTRA_CSV}")
    print(f"candidate_seeds.csv -> {OUT_DIR / 'candidate_seeds.csv'}")
    print(f"group_peak_details.csv -> {OUT_DIR / 'group_peak_details.csv'}")
    print(f"seed_summary.csv -> {OUT_DIR / 'seed_summary.csv'}")
    print(f"scils_import_minimal.csv -> {OUT_DIR / 'scils_import_minimal.csv'}")
    print(f"scils_featurelist_patched.csv -> {OUT_DIR / 'scils_featurelist_patched.csv'}")
    print(f"peak plots -> {OUT_DIR / 'peak_plots'}")
