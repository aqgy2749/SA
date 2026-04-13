import dgl
import torch
import CAST
import os
import numpy as np
import anndata as ad
import scanpy as sc
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from CAST import CAST_MARK
from CAST.visualize import kmeans_plot_multiple
from CAST.models.model_GCNII import Args

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ------------------- 路径配置 -------------------
work_dir = '/p1/data/jtian/SA/embyro'
TRANS1_H5AD = f'{work_dir}/output/demo1_adataTrans1.h5ad'
TRANS2_H5AD = f'{work_dir}/output/demo1_adataTrans2.h5ad'
META1_H5AD  = f'{work_dir}/output/demo1_adataMeta1.h5ad'
META2_H5AD  = f'{work_dir}/output/demo1_adataMeta2.h5ad'

output_path = f'{work_dir}/output_em2_full'
os.makedirs(output_path, exist_ok=True)

# ------------------- GPU 设置 -------------------
GPU_ID = 0 if torch.cuda.is_available() else -1
# GPU_ID = -1  # 强制CPU

# ------------------- 低维维度 -------------------
k = 100

# ------------------- HVG 配置 -------------------
N_HVG = 2000

# ------------------- 背景过滤参数（你主要调这里） -------------------
BG_MIN_COUNTS_Q = 0.20     # 扔掉最低20% counts
BG_MIN_GENES_Q  = 0.20     # 扔掉最低20% genes
BG_KNN_K         = 10      # kNN 第k邻居距离作为密度指标
BG_KNN_DIST_Q    = 0.90    # 扔掉最稀疏10%（kNN距离最大）

# ------------------- 坐标/工具函数 -------------------
def get_xy(adata):
    if ("x" in adata.obs.columns) and ("y" in adata.obs.columns):
        return np.array(adata.obs[["x", "y"]], dtype=np.float32)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=np.float32)
    raise ValueError("找不到坐标：需要 adata.obs['x','y'] 或 adata.obsm['spatial']")

def norm_xy_to(XY, target=2000):
    XY = XY.astype(np.float32)
    XY = XY - XY.min(axis=0)
    mx = XY.max(axis=0)
    s = target / (mx.max() + 1e-8)  # 用最大边缩放，保持比例
    return XY * s

def remove_background_trans(
    adata,
    min_counts_quantile=0.20,
    min_genes_quantile=0.20,
    k=10,
    nn_dist_quantile=0.90,
    copy=True
):
    """
    返回：过滤后的 adata_f, mask_final（在原始 adata 上的bool mask，True=保留）
    """
    ad0 = adata.copy() if copy else adata

    # QC metrics
    sc.pp.calculate_qc_metrics(ad0, inplace=True)
    tc = ad0.obs["total_counts"].to_numpy()
    ng = ad0.obs["n_genes_by_counts"].to_numpy()

    tc_th = np.quantile(tc, min_counts_quantile)
    ng_th = np.quantile(ng, min_genes_quantile)
    mask_qc = (tc >= tc_th) & (ng >= ng_th)

    XY = get_xy(ad0)
    XY_qc = XY[mask_qc]

    # 如果QC后点太少，就只做QC
    if XY_qc.shape[0] <= (k + 1):
        mask_final = mask_qc
    else:
        tree = cKDTree(XY_qc)
        dists, _ = tree.query(XY_qc, k=k+1)   # 包含自己
        kth = dists[:, -1]                    # 到第k邻居距离
        dist_th = np.quantile(kth, nn_dist_quantile)
        mask_dense_qc = kth <= dist_th

        mask_final = np.zeros(ad0.n_obs, dtype=bool)
        idx_qc = np.where(mask_qc)[0]
        mask_final[idx_qc[mask_dense_qc]] = True

    ad_f = ad0[mask_final].copy()
    ad_f.uns["bg_filter"] = dict(
        min_counts_quantile=float(min_counts_quantile),
        min_genes_quantile=float(min_genes_quantile),
        k=int(k),
        nn_dist_quantile=float(nn_dist_quantile),
        kept=int(mask_final.sum()),
        total=int(ad0.n_obs),
        thresholds=dict(
            total_counts=float(tc_th),
            n_genes=float(ng_th),
        )
    )
    return ad_f, mask_final

def plot_bg_filter(ad_before, ad_after, mask_final, out_png, title="trans"):
    """
    三联图：
      1) BEFORE: 全量点，按log1p(total_counts)着色
      2) AFTER:  保留点，按log1p(total_counts)着色
      3) KEPT/REMOVED: removed灰色，kept按log1p(total_counts)着色
    """
    sc.pp.calculate_qc_metrics(ad_before, inplace=True)
    sc.pp.calculate_qc_metrics(ad_after, inplace=True)

    XYb = get_xy(ad_before)
    XYa = get_xy(ad_after)

    cb = np.log1p(ad_before.obs["total_counts"].to_numpy())
    ca = np.log1p(ad_after.obs["total_counts"].to_numpy())

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(XYb[:, 0], XYb[:, 1], c=cb, s=2, linewidths=0)
    plt.gca().set_aspect("equal")
    plt.title(f"{title} BEFORE\ncolor=log1p(total_counts)")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.scatter(XYa[:, 0], XYa[:, 1], c=ca, s=2, linewidths=0)
    plt.gca().set_aspect("equal")
    plt.title(f"{title} AFTER (tissue)\ncolor=log1p(total_counts)")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    # removed
    plt.scatter(XYb[~mask_final, 0], XYb[~mask_final, 1], c="lightgray", s=1, linewidths=0, label="removed")
    # kept colored
    plt.scatter(XYb[mask_final, 0], XYb[mask_final, 1], c=cb[mask_final], s=2, linewidths=0, label="kept")
    plt.gca().set_aspect("equal")
    plt.title(f"{title} KEPT vs REMOVED")
    plt.colorbar()
    plt.legend(markerscale=4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ------------------- 1) 读入 4 个 adata -------------------
adataTrans1 = ad.read_h5ad(TRANS1_H5AD)
adataTrans2 = ad.read_h5ad(TRANS2_H5AD)
adataMeta1  = ad.read_h5ad(META1_H5AD)
adataMeta2  = ad.read_h5ad(META2_H5AD)

print("Trans1:", type(adataTrans1.X), adataTrans1.shape)
print("Trans2:", type(adataTrans2.X), adataTrans2.shape)
print("Meta1 :", type(adataMeta1.X),  adataMeta1.shape)
print("Meta2 :", type(adataMeta2.X),  adataMeta2.shape)

# ------------------- 1.5) trans 去背景 + 可视化 -------------------
adataTrans1_f, mask1 = remove_background_trans(
    adataTrans1,
    min_counts_quantile=BG_MIN_COUNTS_Q,
    min_genes_quantile=BG_MIN_GENES_Q,
    k=BG_KNN_K,
    nn_dist_quantile=BG_KNN_DIST_Q
)
adataTrans2_f, mask2 = remove_background_trans(
    adataTrans2,
    min_counts_quantile=BG_MIN_COUNTS_Q,
    min_genes_quantile=BG_MIN_GENES_Q,
    k=BG_KNN_K,
    nn_dist_quantile=BG_KNN_DIST_Q
)

print(f"[BG] Trans1: {adataTrans1.shape} -> {adataTrans1_f.shape} kept={mask1.sum()}/{len(mask1)}")
print(f"[BG] Trans2: {adataTrans2.shape} -> {adataTrans2_f.shape} kept={mask2.sum()}/{len(mask2)}")

plot_bg_filter(
    adataTrans1, adataTrans1_f, mask1,
    out_png=os.path.join(output_path, "trans1_bgfilter_totalcounts.png"),
    title="trans1"
)
plot_bg_filter(
    adataTrans2, adataTrans2_f, mask2,
    out_png=os.path.join(output_path, "trans2_bgfilter_totalcounts.png"),
    title="trans2"
)

# 用过滤后的替换原始 trans（后续全流程自动用 tissue）
adataTrans1 = adataTrans1_f
adataTrans2 = adataTrans2_f

# 可选：保存过滤后的 trans，方便复现/下次直接读
adataTrans1.write_h5ad(f'{work_dir}/output/demo1_adataTrans1_tissue.h5ad')
adataTrans2.write_h5ad(f'{work_dir}/output/demo1_adataTrans2_tissue.h5ad')

# ------------------- 2) 坐标：缩放到统一范围（0~2000） -------------------
coords_raw = {
    "trans1": norm_xy_to(get_xy(adataTrans1), target=2000),
    "trans2": norm_xy_to(get_xy(adataTrans2), target=2000),
    "meta1":  norm_xy_to(get_xy(adataMeta1),  target=2000),
    "meta2":  norm_xy_to(get_xy(adataMeta2),  target=2000),
}

print("trans1 xy min/max:", coords_raw["trans1"].min(0), coords_raw["trans1"].max(0))
print("meta1  xy min/max:", coords_raw["meta1"].min(0),  coords_raw["meta1"].max(0))

# ------------------- 3) trans：选 HVG 2000 后再做 SVD；meta：PCA -------------------
def select_hvg(adata, n_hvg=2000, flavor="seurat"):
    adata_tmp = adata.copy()
    sc.pp.highly_variable_genes(
        adata_tmp,
        n_top_genes=n_hvg,
        flavor=flavor,
        inplace=True
    )
    hvgs = adata_tmp.var["highly_variable"].values
    if hvgs.sum() < min(n_hvg, adata.n_vars):
        idx = np.arange(min(n_hvg, adata.n_vars))
        hvgs = np.zeros(adata.n_vars, dtype=bool)
        hvgs[idx] = True
    return hvgs

hvg_trans1 = select_hvg(adataTrans1, n_hvg=N_HVG)
hvg_trans2 = select_hvg(adataTrans2, n_hvg=N_HVG)

def to_low_dim(adata, k, mode="trans", hvg_mask=None):
    X = adata.X
    if (mode == "trans") and (hvg_mask is not None):
        X = X[:, hvg_mask]  # 只取 HVG 2000

    if mode == "trans":
        svd = TruncatedSVD(n_components=k, random_state=0)
        Z = svd.fit_transform(X)  # (n_obs, k)
    else:
        if issparse(X):
            X = X.toarray()
        pca = PCA(n_components=k, random_state=0)
        Z = pca.fit_transform(X.astype(np.float32))

    Z = StandardScaler().fit_transform(Z).astype(np.float32)
    return Z

Z_trans1 = to_low_dim(adataTrans1, k, mode="trans", hvg_mask=hvg_trans1)
Z_trans2 = to_low_dim(adataTrans2, k, mode="trans", hvg_mask=hvg_trans2)
Z_meta1  = to_low_dim(adataMeta1,  k, mode="meta")
Z_meta2  = to_low_dim(adataMeta2,  k, mode="meta")

exp_dict = {
    "trans1": Z_trans1,
    "trans2": Z_trans2,
    "meta1":  Z_meta1,
    "meta2":  Z_meta2,
}
print("exp_dict shapes:", {k_: v.shape for k_, v in exp_dict.items()})

# ------------------- 4) CAST.MARK -------------------
args = Args(
    dataname='embyro_4samples_lowdim_full_bgfiltered',
    gpu=GPU_ID,
    epochs=1000,
    lr1=1e-3,
    wd1=0,
    lambd=1e-3,
    n_layers=9,
    der=0.5,
    dfr=0.3,
    use_encoder=False,
    encoder_dim=k,
)

embed_dict = CAST_MARK(coords_raw, exp_dict, output_path, args=args)
torch.save(embed_dict, os.path.join(output_path, "embed_dict_mark_4samples.pt"))

samples = ["trans1", "trans2", "meta1", "meta2"]
kmeans_plot_multiple(
    embed_dict, samples, coords_raw,
    'demo4_full_bgfiltered', output_path,
    k=20, dot_size=10, minibatch=False
)

# ------------------- 5) CAST.STACK：全量对齐（不做 ROI） -------------------
def run_stack_full(query, ref, iters=1500, bleeding=80, iterations_bs=0, tag=None):
    graph_list = [query, ref]
    params_dist = CAST.reg_params(
        dataname=graph_list[0],
        gpu=GPU_ID,
        diff_step=5,
        iterations=iters,
        dist_penalty1=0.1,
        bleeding=int(bleeding),
        d_list=[3, 2, 1, 1/2, 1/3],
        attention_params=[None, 3, 1, 0],
        dist_penalty2=[0],
        alpha_basis_bs=[500],
        meshsize=[8],
        iterations_bs=[iterations_bs],
        attention_params_bs=[[None, 3, 1, 0]],
        mesh_weight=[None],
    )
    params_dist.alpha_basis = torch.Tensor([1/1000, 1/1000, 1/50, 5, 5]).reshape(5, 1).to(params_dist.device)

    out = CAST.CAST_STACK(
        coords_raw=coords_raw,
        embed_dict=embed_dict,
        output_path=output_path,
        graph_list=graph_list,
        params_dist=params_dist,
        sub_node_idxs=None,     # 全量（不做 ROI）
        if_embed_sub=False,     # 全量 embedding/corr
        load_affine_ckpt=False,
        save_affine_ckpt=True,
        affine_ckpt_path=None,
    )
    if tag is None:
        tag = f"{query}_to_{ref}_full"
    torch.save(out, os.path.join(output_path, f"coords_final_{tag}.pt"))
    return out

coords_final_1 = run_stack_full("trans1", "meta1", iters=1500, bleeding=80, iterations_bs=0, tag="trans1_to_meta1_full_bgfiltered")
coords_final_2 = run_stack_full("trans2", "meta2", iters=1500, bleeding=80, iterations_bs=0, tag="trans2_to_meta2_full_bgfiltered")

print("✅ 完成：trans去背景 + HVG(2000)+低维(k)+MARK+FULL-STACK(全量对齐输出)")
print("✅ trans1/trans2 背景过滤图已保存到：")
print("   ", os.path.join(output_path, "trans1_bgfilter_totalcounts.png"))
print("   ", os.path.join(output_path, "trans2_bgfilter_totalcounts.png"))
