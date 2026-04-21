from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser


@dataclass
class MeanSpectraConfig:
    inputs: Sequence[str]
    labels: Sequence[str]
    out: str
    force_processed: bool = False
    mz_step: float = 0.001
    detect_n_check: int = 20
    overlap_n_check: int = 50
    rtol: float = 1e-8
    atol: float = 1e-12
    processed_projection: str = "interp"  # 'interp' or 'bin'


def _sample_indices(n_total: int, n_check: int) -> np.ndarray:
    n = min(n_total, n_check)
    if n <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, n_total - 1, n, dtype=int))


def detect_continuous(
    parser: ImzMLParser,
    n_check: int = 20,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> bool:
    idxs = _sample_indices(len(parser.coordinates), n_check)
    if len(idxs) == 0:
        return False

    mz0, _ = parser.getspectrum(int(idxs[0]))
    for i in idxs[1:]:
        mzi, _ = parser.getspectrum(int(i))
        if len(mzi) != len(mz0):
            return False
        if not np.allclose(mzi, mz0, rtol=rtol, atol=atol):
            return False
    return True


def average_imzml_continuous(
    parser: ImzMLParser,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    mz_ref, inten_ref = parser.getspectrum(0)
    acc = np.zeros_like(np.asarray(inten_ref, dtype=np.float64), dtype=np.float64)

    for i in range(len(parser.coordinates)):
        mz, inten = parser.getspectrum(i)
        if len(mz) != len(mz_ref) or not np.allclose(mz, mz_ref, rtol=rtol, atol=atol):
            raise ValueError("检测到该文件并非严格 continuous，不能直接平均。")
        acc += np.asarray(inten, dtype=np.float64)

    mean_inten = acc / len(parser.coordinates)
    return np.asarray(mz_ref, dtype=np.float64), mean_inten


def estimate_overlap_mz_range(parser: ImzMLParser, n_check: int = 50) -> Tuple[float, float]:
    idxs = _sample_indices(len(parser.coordinates), n_check)
    if len(idxs) == 0:
        raise ValueError("该 imzML 没有可用谱。")

    mins: List[float] = []
    maxs: List[float] = []
    for i in idxs:
        mz, _ = parser.getspectrum(int(i))
        if len(mz) == 0:
            continue
        mins.append(float(np.min(mz)))
        maxs.append(float(np.max(mz)))

    if not mins or not maxs:
        raise ValueError("无法从该 imzML 中估计公共 m/z 范围。")
    return float(np.max(mins)), float(np.min(maxs))


def _build_common_mz(mz_min: float, mz_max: float, mz_step: float) -> np.ndarray:
    if mz_step <= 0:
        raise ValueError("mz_step 必须大于 0。")
    if mz_max <= mz_min:
        raise ValueError("公共 m/z 范围为空。")
    return np.arange(mz_min, mz_max + mz_step / 2, mz_step, dtype=np.float64)


def average_imzml_processed_interp(
    parser: ImzMLParser,
    mz_min: float,
    mz_max: float,
    mz_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SCiLS-like 公共坐标轴均值谱：
    先把每条谱插值到统一 common_mz，再逐点平均。
    没有覆盖到的点按 0 处理。
    """
    common_mz = _build_common_mz(mz_min, mz_max, mz_step)
    acc = np.zeros_like(common_mz, dtype=np.float64)

    n_spec = len(parser.coordinates)
    if n_spec == 0:
        raise ValueError("该 imzML 中没有谱。")

    for i in range(n_spec):
        mz, inten = parser.getspectrum(i)
        mz = np.asarray(mz, dtype=np.float64)
        inten = np.asarray(inten, dtype=np.float64)
        interp_inten = np.interp(common_mz, mz, inten, left=0.0, right=0.0)
        acc += interp_inten

    mean_inten = acc / n_spec
    return common_mz, mean_inten


def average_imzml_processed_bin(
    parser: ImzMLParser,
    mz_min: float,
    mz_max: float,
    mz_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    common_mz = _build_common_mz(mz_min, mz_max, mz_step)
    acc = np.zeros_like(common_mz, dtype=np.float64)

    n_spec = len(parser.coordinates)
    if n_spec == 0:
        raise ValueError("该 imzML 中没有谱。")

    for i in range(n_spec):
        mz, inten = parser.getspectrum(i)
        mz = np.asarray(mz, dtype=np.float64)
        inten = np.asarray(inten, dtype=np.float64)

        mask = np.isfinite(mz) & np.isfinite(inten) & (mz >= mz_min) & (mz <= mz_max)
        if not np.any(mask):
            continue
        mz = mz[mask]
        inten = inten[mask]

        idx = np.rint((mz - mz_min) / mz_step).astype(np.int64)
        valid = (idx >= 0) & (idx < len(common_mz))
        idx = idx[valid]
        inten = inten[valid]
        if len(idx) == 0:
            continue
        np.add.at(acc, idx, inten)

    mean_inten = acc / n_spec
    return common_mz, mean_inten


def build_mean_spectra_csv(config: MeanSpectraConfig) -> pd.DataFrame:
    input_paths = [Path(x) for x in config.inputs]
    labels = list(config.labels)

    if len(input_paths) != len(labels):
        raise ValueError("inputs 和 labels 数量必须一致。")
    if len(input_paths) == 0:
        raise ValueError("至少需要 1 个 imzML 文件。")

    parsers = [ImzMLParser(str(p)) for p in input_paths]

    all_continuous = (
        all(detect_continuous(p, n_check=config.detect_n_check, rtol=config.rtol, atol=config.atol) for p in parsers)
        and (not config.force_processed)
    )

    out_df: pd.DataFrame | None = None

    if all_continuous:
        for path, label, parser in zip(input_paths, labels, parsers):
            mz, mean_inten = average_imzml_continuous(parser, rtol=config.rtol, atol=config.atol)
            if out_df is None:
                out_df = pd.DataFrame({"mz": mz})
            else:
                if len(out_df) != len(mz) or not np.allclose(out_df["mz"].to_numpy(), mz, rtol=config.rtol, atol=config.atol):
                    raise ValueError(f"{path.name} 的 m/z 轴与前面文件不一致。")
            out_df[label] = mean_inten
    else:
        ranges = [estimate_overlap_mz_range(p, n_check=config.overlap_n_check) for p in parsers]
        mz_min = max(r[0] for r in ranges)
        mz_max = min(r[1] for r in ranges)
        if mz_max <= mz_min:
            raise ValueError("多个 ROI imzML 的公共 m/z 范围为空，无法统一到公共坐标轴。")

        projector = average_imzml_processed_interp
        if config.processed_projection == "bin":
            projector = average_imzml_processed_bin
        elif config.processed_projection != "interp":
            raise ValueError("processed_projection 只能是 'interp' 或 'bin'。")

        for path, label, parser in zip(input_paths, labels, parsers):
            mz, mean_inten = projector(
                parser,
                mz_min=mz_min,
                mz_max=mz_max,
                mz_step=config.mz_step,
            )
            if out_df is None:
                out_df = pd.DataFrame({"mz": mz})
            else:
                if len(out_df) != len(mz) or not np.allclose(out_df["mz"].to_numpy(), mz, rtol=config.rtol, atol=config.atol):
                    raise ValueError(f"{path.name} 生成后的 m/z 网格与前面文件不一致。")
            out_df[label] = mean_inten

    assert out_df is not None
    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df
