from pathlib import Path

from scils_overview_pkg import OverviewBuildConfig, OverviewPipelineConfig, run_overview_pipeline
from scils_peak_pkg import PeakWindowConfig


PROJECT_DIR = Path("/p2/zulab/jtian/data/SA/03_removeLow-qualityPeaks/ScilsPeakFilter")
GROUPS = ["ctrl1", "ctrl2", "dn1", "dn2"]
FEATURELIST = PROJECT_DIR / "2326featureList.csv"
RUN_PLOTS = False


for group in GROUPS:
    overview_dir = PROJECT_DIR / "input" / "overviewSpectra" / group
    out_dir = PROJECT_DIR / f"output_run_overview_pipeline_{group}"
    mean_spectra_csv = out_dir / "mean_spectra_from_overview.csv"

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output dir: {out_dir}")
    else:
        print(f"Output dir exists: {out_dir}")

    overview_cfg = OverviewBuildConfig(
        input_dir=str(overview_dir),
        out_csv=str(mean_spectra_csv),
        file_pattern="*-OverviewSpectra.csv",
        align_how="inner",
    )

    peak_cfg = PeakWindowConfig(
        featurelist=str(FEATURELIST),
        mean_spectra=str(mean_spectra_csv),
        outdir=str(out_dir),
        seed_source="featurelist",
        do_smooth=False,
        do_baseline=False,
        smooth_window=11,
        smooth_polyorder=3,
        baseline_window=101,
        search_ppm=50.0,
        prominence_frac=0.05,
        snr_threshold=3.0,
        fwhm_min_ppm=3.0,
        fwhm_max_ppm=500.0,
        valley_ratio_max=0.50,
        secondary_peak_ratio_max=0.70,
        base_rel_height=0.10,
        require_all_groups=False,
        min_independent_groups=4,
        min_candidate_groups=4,
        require_first_group_independent=True,
        max_interval_half_width_da=50.0,
    )

    pipeline_cfg = OverviewPipelineConfig(
        overview=overview_cfg,
        peak_windows=peak_cfg,
        featurelist=str(FEATURELIST),
        plot_overview_png=str(out_dir / "overview_from_scils_export.png") if RUN_PLOTS else None,
        plot_all_features=RUN_PLOTS,
        peak_plot_dir=str(out_dir / "peak_plots") if RUN_PLOTS else None,
    )

    run_overview_pipeline(pipeline_cfg)

    scils_import_csv = out_dir / "scils_import_minimal.csv"
    with open(scils_import_csv, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"{scils_import_csv} 是空文件。")

    header = lines[0]
    data_lines = lines[1:]
    split_n = (len(data_lines) + 1) // 2

    part1_csv = scils_import_csv.with_name(f"{scils_import_csv.stem}_part1.csv")
    part2_csv = scils_import_csv.with_name(f"{scils_import_csv.stem}_part2.csv")

    with open(part1_csv, "w", encoding="utf-8-sig") as f:
        f.write(header)
        f.writelines(data_lines[:split_n])

    with open(part2_csv, "w", encoding="utf-8-sig") as f:
        f.write(header)
        f.writelines(data_lines[split_n:])

    print(f"Done: {group}")
    print(f"mean_spectra_from_overview.csv -> {mean_spectra_csv}")
    print(f"group_peak_details.csv -> {out_dir / 'group_peak_details.csv'}")
    print(f"seed_summary.csv -> {out_dir / 'seed_summary.csv'}")
    print(f"scils_import_minimal.csv -> {scils_import_csv}")
    print(f"scils_import_minimal_part1.csv -> {part1_csv}")
    print(f"scils_import_minimal_part2.csv -> {part2_csv}")
    print(f"scils_featurelist_patched.csv -> {out_dir / 'scils_featurelist_patched.csv'}")
    if RUN_PLOTS:
        print(f"peak plots -> {out_dir / 'peak_plots'}")
