import CAST
import scanpy as sc
import os
import numpy as np
import warnings
import dgl
import torch
import CAST
import os
import numpy as np
import anndata as ad
import scanpy as sc
import warnings
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
with open(os.path.join('/p2/zulab/jtian/data/SA/06_calculateConcentration/output_neighborsBanksy/', 'display_frames.pkl'), 'rb') as f:
    display_frames = pickle.load(f)
# =========================================
# 1. 基础函数：邻域图与平滑
# =========================================

def _compute_knn_indices_and_distances(xy, k=15):
    """
    xy: shape (n, 2)
    返回每个点最近的 k 个邻居（不含自己）的索引和距离
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]

    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    # 不把自己作为邻居
    np.fill_diagonal(dist, np.inf)

    k_eff = min(k, max(n - 1, 1))
    nbr_idx = np.argsort(dist, axis=1)[:, :k_eff]
    nbr_dist = np.take_along_axis(dist, nbr_idx, axis=1)

    return nbr_idx, nbr_dist


def _make_neighbor_weights(nbr_dist, decay='scaled_gaussian', eps=1e-12):
    """
    根据邻居距离构建权重
    decay:
    - 'scaled_gaussian'
    - 'uniform'
    - 'reciprocal'
    """
    d = np.asarray(nbr_dist, dtype=float)

    if decay == 'uniform':
        w = np.ones_like(d, dtype=float)

    elif decay == 'reciprocal':
        w = 1.0 / np.maximum(d, eps)

    elif decay == 'scaled_gaussian':
        # 每个点用第 k 个邻居距离作为局部 sigma
        sigma = d[:, -1][:, None]
        sigma = np.maximum(sigma, eps)
        w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))

    else:
        raise ValueError("decay 只能是 'scaled_gaussian', 'uniform', 'reciprocal'")

    # 行归一化
    row_sum = np.sum(w, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, eps)
    w = w / row_sum
    return w


def _neighbor_mean_matrix(X, nbr_idx, W):
    """
    对 X 计算邻域均值矩阵 M
    X: (n, p)
    nbr_idx: (n, k)
    W: (n, k)

    自动忽略 NaN，并对每一列按有效邻居重新归一化
    """
    X = np.asarray(X, dtype=float)
    X_neigh = X[nbr_idx, :]      # (n, k, p)
    valid = np.isfinite(X_neigh)

    numerator = np.nansum(X_neigh * W[:, :, None], axis=1)
    denominator = np.sum(W[:, :, None] * valid, axis=1)

    M = np.full_like(numerator, np.nan, dtype=float)
    good = denominator > 0
    M[good] = numerator[good] / denominator[good]
    return M


def _split_feature_name(col):
    """
    把 '57.0346-45' 拆成 ('57.0346', 45)
    """
    if '-' not in col:
        return None, None
    mz, sample = col.rsplit('-', 1)
    if not sample.isdigit():
        return None, None
    return mz, int(sample)


def _auto_feature_cols(df):
    """
    自动识别所有 mz-sample 列
    """
    cols = []
    for c in df.columns:
        mz, sample = _split_feature_name(c)
        if mz is not None:
            cols.append(c)
    return cols


def banksy_like_smooth_display0(
    df0,
    coord_cols=('0-original-x', '0-original-y'),
    feature_cols=None,
    k=15,
    lambda_=0.2,
    decay='scaled_gaussian',
    use_log1p=True,
    preserve_missing=True
):
    """
    对 display_frames[0] 做 BANKSY-inspired 微环境平滑

    X* = (1-lambda) X + lambda M

    参数
    ----
    df0 : display_frames[0]
    coord_cols : 用于建邻域图的坐标列
    feature_cols : 要平滑的列；None 表示自动识别所有 mz-sample 列
    k : 邻居数
    lambda_ : 微环境权重
    decay : 权重衰减方式
    use_log1p : 是否先 log1p 再平滑
    preserve_missing : 原始缺失是否保持缺失

    返回
    ----
    smoothed_df : 平滑后的表
    neighbor_mean_df : 邻域均值表 M
    weights_df : 邻居与权重明细
    """
    if not (0 <= lambda_ <= 1):
        raise ValueError("lambda_ 必须在 0 到 1 之间")

    out = df0.copy()
    xcol, ycol = coord_cols

    out[xcol] = pd.to_numeric(out[xcol], errors='coerce')
    out[ycol] = pd.to_numeric(out[ycol], errors='coerce')

    if feature_cols is None:
        feature_cols = _auto_feature_cols(out)

    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')

    valid_mask = out[[xcol, ycol]].notna().all(axis=1)
    valid_index = out.index[valid_mask].to_numpy()

    if len(valid_index) < 2:
        raise ValueError("有效坐标点少于 2 个，无法建邻域图")

    xy = out.loc[valid_mask, [xcol, ycol]].to_numpy(dtype=float)
    X_raw = out.loc[valid_mask, feature_cols].to_numpy(dtype=float)

    if use_log1p:
        if np.nanmin(X_raw) < 0:
            raise ValueError("检测到负 intensity，不能直接做 log1p；请先检查数据或设 use_log1p=False")
        X = np.log1p(X_raw)
    else:
        X = X_raw.copy()

    nbr_idx, nbr_dist = _compute_knn_indices_and_distances(xy, k=k)
    W = _make_neighbor_weights(nbr_dist, decay=decay)
    M = _neighbor_mean_matrix(X, nbr_idx, W)

    # 平滑
    X_smooth = np.where(
        np.isfinite(X) & np.isfinite(M),
        (1 - lambda_) * X + lambda_ * M,
        np.where(np.isfinite(X), X, M)
    )

    if preserve_missing:
        X_smooth[~np.isfinite(X)] = np.nan

    # 变回原始尺度
    if use_log1p:
        X_smooth_raw = np.expm1(X_smooth)
        M_raw = np.expm1(M)
    else:
        X_smooth_raw = X_smooth
        M_raw = M

    smoothed_df = out.copy()
    smoothed_df.loc[valid_mask, feature_cols] = X_smooth_raw

    neighbor_mean_df = out.copy()
    neighbor_mean_df.loc[:, feature_cols] = np.nan
    neighbor_mean_df.loc[valid_mask, feature_cols] = M_raw

    # 邻居权重表
    weight_records = []
    for i in range(len(valid_index)):
        center_row = int(valid_index[i])
        for j in range(nbr_idx.shape[1]):
            neigh_local = int(nbr_idx[i, j])
            neigh_row = int(valid_index[neigh_local])
            weight_records.append({
                'center_row': center_row,
                'neighbor_rank': j + 1,
                'neighbor_row': neigh_row,
                'distance': float(nbr_dist[i, j]),
                'weight': float(W[i, j]),
            })
    weights_df = pd.DataFrame(weight_records)

    return smoothed_df, neighbor_mean_df, weights_df


# =========================================
# 2. 标准曲线拟合：每一行、每一个 mz
# =========================================

def _safe_line_fit_and_corr(x, y, min_n=3):
    """
    对一组 (x, y) 做线性拟合并返回 Pearson / R² / slope / intercept

    x: 样本编号，例如 [0,15,30,45,60]
    y: 同一行、同一 mz 在不同样本上的 intensity
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]
    n = len(x2)

    if n < min_n:
        return {
            'n': n,
            'pearson_r': np.nan,
            'r2': np.nan,
            'slope': np.nan,
            'intercept': np.nan
        }

    # x 或 y 无方差时，相关性和回归都没有意义
    if np.nanstd(x2) == 0 or np.nanstd(y2) == 0:
        return {
            'n': n,
            'pearson_r': np.nan,
            'r2': np.nan,
            'slope': np.nan,
            'intercept': np.nan
        }

    # Pearson
    r = np.corrcoef(x2, y2)[0, 1]

    # 一元线性回归 y = slope * x + intercept
    slope, intercept = np.polyfit(x2, y2, 1)

    # R²
    y_hat = slope * x2 + intercept
    ss_res = np.sum((y2 - y_hat) ** 2)
    ss_tot = np.sum((y2 - np.mean(y2)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    return {
        'n': n,
        'pearson_r': float(r),
        'r2': float(r2),
        'slope': float(slope),
        'intercept': float(intercept)
    }


def evaluate_standard_curve_before_after(
    df_before,
    df_after,
    samples=(0, 15, 30, 45, 60),
    mz_list=None,
    min_n=3
):
    """
    对每个点位、每个 mz：
    把 mz-0, mz-15, mz-30, mz-45, mz-60 视为 y，
    把 0,15,30,45,60 视为 x，
    分别在平滑前后做线性拟合和 Pearson 相关性比较。

    返回
    ----
    detail_df : 每一行每一个 mz 的详细结果
    mz_summary_df : 按 mz 汇总
    overall_summary_df : 整体汇总
    """
    df_before = df_before.copy()
    df_after = df_after.copy()

    # 自动识别所有 mz
    if mz_list is None:
        feature_cols = _auto_feature_cols(df_before)
        mz_set = set()
        for c in feature_cols:
            mz, sample = _split_feature_name(c)
            if mz is not None:
                mz_set.add(mz)
        mz_list = sorted(mz_set, key=lambda z: float(z))

    records = []

    for row_idx in df_before.index:
        for mz in mz_list:
            cols = [f'{mz}-{s}' for s in samples if f'{mz}-{s}' in df_before.columns]

            if len(cols) == 0:
                continue

            # x 是样本编号
            x = np.array([int(c.rsplit('-', 1)[1]) for c in cols], dtype=float)

            # y 是这一行各样本的 intensity
            y_before = pd.to_numeric(df_before.loc[row_idx, cols], errors='coerce').to_numpy(dtype=float)
            y_after = pd.to_numeric(df_after.loc[row_idx, cols], errors='coerce').to_numpy(dtype=float)

            fit_before = _safe_line_fit_and_corr(x, y_before, min_n=min_n)
            fit_after = _safe_line_fit_and_corr(x, y_after, min_n=min_n)

            records.append({
                'row': int(row_idx),
                'mz': mz,

                'n_before': fit_before['n'],
                'pearson_before': fit_before['pearson_r'],
                'r2_before': fit_before['r2'],
                'slope_before': fit_before['slope'],
                'intercept_before': fit_before['intercept'],

                'n_after': fit_after['n'],
                'pearson_after': fit_after['pearson_r'],
                'r2_after': fit_after['r2'],
                'slope_after': fit_after['slope'],
                'intercept_after': fit_after['intercept'],

                'delta_pearson': (
                    fit_after['pearson_r'] - fit_before['pearson_r']
                    if pd.notna(fit_before['pearson_r']) and pd.notna(fit_after['pearson_r'])
                    else np.nan
                ),
                'delta_r2': (
                    fit_after['r2'] - fit_before['r2']
                    if pd.notna(fit_before['r2']) and pd.notna(fit_after['r2'])
                    else np.nan
                )
            })

    detail_df = pd.DataFrame(records)

    if len(detail_df) == 0:
        return detail_df, pd.DataFrame(), pd.DataFrame()

    # 按 mz 汇总
    mz_summary_df = (
        detail_df
        .groupby('mz', dropna=False)
        .agg(
            n_rows=('row', 'count'),
            mean_pearson_before=('pearson_before', 'mean'),
            median_pearson_before=('pearson_before', 'median'),
            mean_pearson_after=('pearson_after', 'mean'),
            median_pearson_after=('pearson_after', 'median'),
            mean_delta_pearson=('delta_pearson', 'mean'),
            median_delta_pearson=('delta_pearson', 'median'),

            mean_r2_before=('r2_before', 'mean'),
            median_r2_before=('r2_before', 'median'),
            mean_r2_after=('r2_after', 'mean'),
            median_r2_after=('r2_after', 'median'),
            mean_delta_r2=('delta_r2', 'mean'),
            median_delta_r2=('delta_r2', 'median'),

            improved_pearson_frac=('delta_pearson', lambda s: np.mean(s > 0)),
            improved_r2_frac=('delta_r2', lambda s: np.mean(s > 0)),
        )
        .reset_index()
    )

    # 整体汇总
    overall_summary_df = pd.DataFrame([{
        'n_row_mz_pairs': len(detail_df),

        'mean_pearson_before': detail_df['pearson_before'].mean(),
        'median_pearson_before': detail_df['pearson_before'].median(),
        'mean_pearson_after': detail_df['pearson_after'].mean(),
        'median_pearson_after': detail_df['pearson_after'].median(),
        'mean_delta_pearson': detail_df['delta_pearson'].mean(),
        'median_delta_pearson': detail_df['delta_pearson'].median(),
        'improved_pearson_frac': np.mean(detail_df['delta_pearson'] > 0),

        'mean_r2_before': detail_df['r2_before'].mean(),
        'median_r2_before': detail_df['r2_before'].median(),
        'mean_r2_after': detail_df['r2_after'].mean(),
        'median_r2_after': detail_df['r2_after'].median(),
        'mean_delta_r2': detail_df['delta_r2'].mean(),
        'median_delta_r2': detail_df['delta_r2'].median(),
        'improved_r2_frac': np.mean(detail_df['delta_r2'] > 0),
    }])

    return detail_df, mz_summary_df, overall_summary_df


# =========================================
# 3. 一键主流程
# =========================================

def run_banksy_like_standard_curve_pipeline(
    display_frames,
    key=0,
    coord_cols=('0-original-x', '0-original-y'),
    samples=(0, 15, 30, 45, 60),
    k=15,
    lambda_=0.2,
    decay='scaled_gaussian',
    use_log1p=True,
    preserve_missing=True,
    min_n=3
):
    """
    对 display_frames[0]：
    1) BANKSY-inspired 微环境平滑
    2) 按“每行每个 mz 的标准曲线”比较平滑前后 Pearson / R²
    """
    df0 = display_frames[key].copy()
    feature_cols = _auto_feature_cols(df0)

    smoothed_df, neighbor_mean_df, weights_df = banksy_like_smooth_display0(
        df0=df0,
        coord_cols=coord_cols,
        feature_cols=feature_cols,
        k=k,
        lambda_=lambda_,
        decay=decay,
        use_log1p=use_log1p,
        preserve_missing=preserve_missing
    )

    detail_df, mz_summary_df, overall_summary_df = evaluate_standard_curve_before_after(
        df_before=df0,
        df_after=smoothed_df,
        samples=samples,
        mz_list=None,
        min_n=min_n
    )

    return {
        'original_df': df0,
        'smoothed_df': smoothed_df,
        'neighbor_mean_df': neighbor_mean_df,
        'weights_df': weights_df,
        'detail_df': detail_df,
        'mz_summary_df': mz_summary_df,
        'overall_summary_df': overall_summary_df
    }
# result = run_banksy_like_standard_curve_pipeline(
#     display_frames=display_frames,
#     key=0,
#     coord_cols=('0-original-x', '0-original-y'),
#     samples=(0, 15, 30, 45, 60),
#     k=15,
#     lambda_=0.8,
#     decay='scaled_gaussian',
#     use_log1p=True,
#     preserve_missing=True,
#     min_n=3
# )
# df0_original = result['original_df']
# df0_smoothed = result['smoothed_df']
# df0_neighbor_mean = result['neighbor_mean_df']
# weights_df = result['weights_df']

# detail_df = result['detail_df']                 # 每行每个 mz 的拟合结果
# mz_summary_df = result['mz_summary_df']         # 按 mz 汇总
# overall_summary_df = result['overall_summary_df']   # 全局汇总
out_dir = "/p2/zulab/jtian/data/SA/06_calculateConcentration/output_neighborsBanksy2/"
# os.makedirs(out_dir, exist_ok=True)

# df0_smoothed.to_csv(os.path.join(out_dir, 'display0_smoothed.csv'), index=False)
# df0_neighbor_mean.to_csv(os.path.join(out_dir, 'display0_neighbor_mean.csv'), index=False)
# weights_df.to_csv(os.path.join(out_dir, 'display0_neighbor_weights.csv'), index=False)

# detail_df.to_csv(os.path.join(out_dir, 'standard_curve_detail_before_after.csv'), index=False)
# mz_summary_df.to_csv(os.path.join(out_dir, 'standard_curve_mz_summary_before_after.csv'), index=False)
# overall_summary_df.to_csv(os.path.join(out_dir, 'standard_curve_overall_summary_before_after.csv'), index=False)
# # 阈值
# r2_thr = 0.9
# site_thr = 9478

# # 复制一份，避免改原表
# tmp = detail_df.copy()

# # 标记每个点-代谢物组合是否满足 R2 >= 0.9
# tmp['good_before'] = tmp['r2_before'] >= r2_thr
# tmp['good_after'] = tmp['r2_after'] >= r2_thr

# # ---------------------------------
# # 1) 统计每个代谢物 R2 >= 0.9 的位点数
# # ---------------------------------
# mz_r2_count_df = (
#     tmp.groupby('mz', dropna=False)
#     .agg(
#         n_sites_total=('row', 'count'),
#         n_sites_r2_ge_0p9_before=('good_before', 'sum'),
#         n_sites_r2_ge_0p9_after=('good_after', 'sum')
#     )
#     .reset_index()
# )

# # 也可以算比例，方便看
# mz_r2_count_df['frac_sites_r2_ge_0p9_before'] = (
#     mz_r2_count_df['n_sites_r2_ge_0p9_before'] / mz_r2_count_df['n_sites_total']
# )
# mz_r2_count_df['frac_sites_r2_ge_0p9_after'] = (
#     mz_r2_count_df['n_sites_r2_ge_0p9_after'] / mz_r2_count_df['n_sites_total']
# )

# # ---------------------------------
# # 2) 统计“满足位点数 >= 9478”的代谢物个数
# # ---------------------------------
# num_mz_before = int((mz_r2_count_df['n_sites_r2_ge_0p9_before'] >= site_thr).sum())
# num_mz_after = int((mz_r2_count_df['n_sites_r2_ge_0p9_after'] >= site_thr).sum())

# # 把满足条件的代谢物名字也取出来
# good_mz_before = mz_r2_count_df.loc[
#     mz_r2_count_df['n_sites_r2_ge_0p9_before'] >= site_thr, 'mz'
# ].tolist()

# good_mz_after = mz_r2_count_df.loc[
#     mz_r2_count_df['n_sites_r2_ge_0p9_after'] >= site_thr, 'mz'
# ].tolist()

# # ---------------------------------
# # 3) 整理一个总汇总表
# # ---------------------------------
# summary_threshold_df = pd.DataFrame({
#     'condition': [
#         'before: n_sites_r2>=0.9 >= 9478',
#         'after:  n_sites_r2>=0.9 >= 9478'
#     ],
#     'n_metabolites': [
#         num_mz_before,
#         num_mz_after
#     ]
# })

# print("每个代谢物满足 R2>=0.9 的位点数：")
# print(mz_r2_count_df.head())

# print("\n平滑前，满足“R2>=0.9的位点数 >= 9478”的代谢物个数：")
# print(num_mz_before)

# print("\n平滑后，满足“R2>=0.9的位点数 >= 9478”的代谢物个数：")
# print(num_mz_after)

# print("\n总汇总：")
# print(summary_threshold_df)
# pd.DataFrame({"mz": good_mz_before}).to_csv(
#     os.path.join(out_dir, "good_mz_before.csv"),
#     index=False
# )

# pd.DataFrame({"mz": good_mz_after}).to_csv(
#     os.path.join(out_dir, "good_mz_after.csv"),
#     index=False
# )
df0_smoothed = pd.read_csv(os.path.join(out_dir, "display0_smoothed.csv"))
detail_df = pd.read_csv(os.path.join(out_dir, "standard_curve_detail_before_after.csv"))
good_mz_after = pd.read_csv(os.path.join(out_dir, "good_mz_after.csv"))

good_mz_after_set = set(map(str, good_mz_after['mz']))

good_mz_xintercept_df = detail_df[
    detail_df['mz'].astype(str).isin(good_mz_after_set)
].copy()

good_mz_xintercept_df = good_mz_xintercept_df.loc[
    :, ['row', 'mz', 'r2_after', 'slope_after', 'intercept_after']
].copy()

good_mz_xintercept_df = good_mz_xintercept_df.sort_values(
    by=['mz', 'row'],
    ascending=[True, True]
).reset_index(drop=True)

# 用你指定的稳健写法计算横截距绝对值
good_mz_xintercept_df['x_intercept_abs_after'] = np.where(
    good_mz_xintercept_df['slope_after'].notna() &
    (good_mz_xintercept_df['slope_after'] != 0) &
    good_mz_xintercept_df['intercept_after'].notna(),
    np.abs(-good_mz_xintercept_df['intercept_after'] / good_mz_xintercept_df['slope_after']),
    np.nan
)

coord_cols = ['0-original-x', '0-original-y']
coords = (
    df0_smoothed.loc[good_mz_xintercept_df['row'], coord_cols]
    .reset_index(drop=True)
)

plot_df = pd.concat(
    [good_mz_xintercept_df.reset_index(drop=True), coords],
    axis=1
)

plot_df = plot_df.rename(columns={
    '0-original-x': 'x',
    '0-original-y': 'y',
    'x_intercept_abs_after': 'conc'
})

plot_df = plot_df.copy()
plot_df.index = range(plot_df.shape[0])


# =========================================
# 5. 只保留 r2_after >= 0.9 的点位用于作图
# =========================================

r2_thr = 0.9

plot_df['r2_after'] = pd.to_numeric(plot_df['r2_after'], errors='coerce')
plot_df['x'] = pd.to_numeric(plot_df['x'], errors='coerce')
plot_df['y'] = pd.to_numeric(plot_df['y'], errors='coerce')
plot_df['conc'] = pd.to_numeric(plot_df['conc'], errors='coerce')

plot_df = plot_df[plot_df['r2_after'] >= r2_thr].copy()
plot_df.index = range(plot_df.shape[0])


# =========================================
# 6. 画热图，并在标题中加参与作图点数
# =========================================

out_dir = '/p2/zulab/jtian/data/SA/06_calculateConcentration/output_neighborsBanksy2/heatmaps_log_clip_5-95_r2ge0p9'
os.makedirs(out_dir, exist_ok=True)

low_q = 5
high_q = 95

mz_list = sorted(plot_df['mz'].dropna().unique())

for mz in mz_list:
    tmp = plot_df[plot_df['mz'] == mz].copy()

    tmp['x'] = pd.to_numeric(tmp['x'], errors='coerce')
    tmp['y'] = pd.to_numeric(tmp['y'], errors='coerce')
    tmp['conc'] = pd.to_numeric(tmp['conc'], errors='coerce')
    tmp['r2_after'] = pd.to_numeric(tmp['r2_after'], errors='coerce')

    tmp = tmp.dropna(subset=['x', 'y', 'conc', 'r2_after'])

    # 只用正值参与显示和定色标
    tmp = tmp[tmp['conc'] > 0]

    if tmp.empty:
        continue

    # 参与作图的点数
    n_points = len(tmp)

    # 用当前 mz 自己的值确定色标范围
    log_vals = np.log1p(tmp['conc'])
    vmin = np.nanpercentile(log_vals, low_q)
    vmax = np.nanpercentile(log_vals, high_q)

    if pd.isna(vmin) or pd.isna(vmax) or vmin >= vmax:
        continue

    tmp['conc_plot'] = np.log1p(tmp['conc'])
    tmp['conc_plot'] = tmp['conc_plot'].clip(lower=vmin, upper=vmax)

    x_unique = np.sort(tmp['x'].unique())
    y_unique = np.sort(tmp['y'].unique())

    heat = tmp.pivot_table(
        index='y',
        columns='x',
        values='conc_plot',
        aggfunc='mean'
    )
    heat = heat.reindex(index=y_unique, columns=x_unique)

    dx = np.min(np.diff(x_unique)) if len(x_unique) > 1 else 1
    dy = np.min(np.diff(y_unique)) if len(y_unique) > 1 else 1

    Z = np.ma.masked_invalid(heat.values)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        Z,
        origin='lower',
        aspect='equal',
        extent=[
            x_unique.min() - dx / 2, x_unique.max() + dx / 2,
            y_unique.min() - dy / 2, y_unique.max() + dy / 2
        ],
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(im, shrink=0.8, label='log1p(concentration)')
    plt.title(f'Metabolite {mz} heatmap\nvalid pixels: {n_points}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    save_path = os.path.join(out_dir, f'{mz}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

print('画完了')








