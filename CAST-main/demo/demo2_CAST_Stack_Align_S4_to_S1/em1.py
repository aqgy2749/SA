import os
import numpy as np
import torch
import scanpy as sc
import CAST
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
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
from sklearn.decomposition import PCA
from CAST.visualize import kmeans_plot_multiple
from CAST import CAST_STACK
from CAST.CAST_Stack import reg_params
from CAST.models.model_GCNII import Args

# ------------------- 配置 -------------------
k = 50
work_dir = "/p1/data/jtian/SA/embyro"
output_path = f"{work_dir}/output_em1"
os.makedirs(output_path, exist_ok=True)

device_gpu = 0 if torch.cuda.is_available() else -1

# ROI（基于“缩放后的坐标系”）
ROI_CENTER = np.array([700.0, 1000.0], dtype=np.float32)
ROI_RADIUS = 100.0
MIN_PTS = 500

# ------------------- 1) 读入h5ad -------------------
adataTrans1 = sc.read_h5ad("/p1/data/jtian/SA/embyro/output/demo1_adataTrans1.h5ad")
adataMeta1  = sc.read_h5ad("/p1/data/jtian/SA/embyro/output/demo1_adataMeta1.h5ad")

print("Trans1:", type(adataTrans1.X), adataTrans1.X.shape)
print("Meta1 :", type(adataMeta1.X),  adataMeta1.X.shape)

# ------------------- 2) 取坐标 + 缩放到统一范围 -------------------
def get_xy(adata):
    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        return adata.obs[["x", "y"]].to_numpy().astype(np.float32)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"]).astype(np.float32)
    raise ValueError("找不到坐标：需要 adata.obs['x','y'] 或 adata.obsm['spatial']")

XY_trans = get_xy(adataTrans1)
XY_meta  = get_xy(adataMeta1)

def norm_xy_to(XY, target=2000):
    XY = XY.astype(np.float32)
    XY = XY - XY.min(axis=0)
    mx = XY.max(axis=0)
    s = target / (mx.max() + 1e-8)  # 用最大边缩放，保持长宽比
    XY = XY * s
    return XY

coords_raw = {
    "trans": norm_xy_to(XY_trans, target=2000),
    "meta":  norm_xy_to(XY_meta,  target=2000),
}

print("XY_trans min/max:", coords_raw["trans"].min(0), coords_raw["trans"].max(0))
print("XY_meta  min/max:", coords_raw["meta"].min(0),  coords_raw["meta"].max(0))

# ------------------- 2.5) 只用 ROI 点参与优化：构造 sub_node_idxs（但不裁剪全量数据） -------------------
def circle_mask(XY, center, radius):
    center = center.astype(np.float32)
    d2 = np.sum((XY - center) ** 2, axis=1)
    return d2 <= (radius ** 2)

mask_trans = circle_mask(coords_raw["trans"], ROI_CENTER, ROI_RADIUS)
mask_meta  = circle_mask(coords_raw["meta"],  ROI_CENTER, ROI_RADIUS)

n_trans_roi = int(mask_trans.sum())
n_meta_roi  = int(mask_meta.sum())
print(f"ROI counts - trans: {n_trans_roi}, meta: {n_meta_roi}")

if n_trans_roi < MIN_PTS or n_meta_roi < MIN_PTS:
    print("⚠️ ROI 点数偏少，可能导致对齐不稳定：建议增大 ROI_RADIUS 或降低 MIN_PTS。")

sub_node_idxs = {
    "trans": mask_trans,
    "meta":  mask_meta,
}

# ------------------- 3) 各自降维到同一个 k 维 -------------------
svd = TruncatedSVD(n_components=k, random_state=0)
Z_trans = svd.fit_transform(adataTrans1.X)  # (n_trans, k) dense

pca = PCA(n_components=k, random_state=0)
X_meta = adataMeta1.X
if issparse(X_meta):
    X_meta = X_meta.toarray()
Z_meta = pca.fit_transform(X_meta.astype(np.float32))  # (n_meta, k)

# ------------------- 4) 低维再标准化（建议保留） -------------------
Z_trans = StandardScaler().fit_transform(Z_trans).astype(np.float32)
Z_meta  = StandardScaler().fit_transform(Z_meta).astype(np.float32)

# ------------------- 5) 全量 embed_dict（不裁剪） -------------------
embed_dict = {
    "trans": torch.tensor(Z_trans, dtype=torch.float32),
    "meta":  torch.tensor(Z_meta,  dtype=torch.float32),
}
torch.save(embed_dict, os.path.join(output_path, "embed_dict_full.pt"))
samples = np.array(["trans", "meta"], dtype=object)
kmeans_plot_multiple(embed_dict, samples, coords_raw, 'em1_pca', output_path, k=20, dot_size=10, minibatch=False)
# ------------------- 6) CAST.STACK：只用 ROI 点算 corr/loss，但把变换应用到全量坐标 -------------------
graph_list = ["trans", "meta"]

params_dist = CAST.reg_params(
    dataname=graph_list[0],
    gpu=device_gpu,
    diff_step=5,
    iterations=1500,
    dist_penalty1=0.1,
    bleeding=20,
    d_list=[3, 2, 1, 1/2, 1/3],
    attention_params=[None, 3, 1, 0],
    dist_penalty2=[0],
    alpha_basis_bs=[500],
    meshsize=[8],
    iterations_bs=[400],
    attention_params_bs=[[None, 3, 1, 0]],
    mesh_weight=[None],
)
params_dist.alpha_basis = torch.tensor([1/1000, 1/1000, 1/50, 5, 5], dtype=torch.float32).reshape(5,1).to(params_dist.device)

coords_final_full = CAST.CAST_STACK(
    coords_raw=coords_raw,          # ✅ 传全量坐标
    embed_dict=embed_dict,          # ✅ 传全量 embedding
    output_path=output_path,
    graph_list=graph_list,
    params_dist=params_dist,
    sub_node_idxs=sub_node_idxs,    # ✅ 只用 ROI 点参与优化
    if_embed_sub=True,              # ✅ corr_dist 也只在 ROI 上算（更省显存/更“只看ROI”）
    load_affine_ckpt=False,
    save_affine_ckpt=True,
    affine_ckpt_path=None,
)

torch.save(coords_final_full, os.path.join(output_path, "coords_final_full_warp.pt"))
print("完成对齐（ROI拟合，但全量warp）。keys:", coords_final_full.keys())
print("trans aligned coords shape:", coords_final_full["trans"].shape)  # ✅ 全量 n_trans
print("meta coords shape:", coords_final_full["meta"].shape)            # ✅ 全量 n_meta






# ------------------- 配置 -------------------
k = 50
work_dir = "/p1/data/jtian/SA/embyro"
output_path = f"{work_dir}/output_em1"
os.makedirs(output_path, exist_ok=True)

device_gpu = 0 if torch.cuda.is_available() else -1









# ROI（基于“缩放后的坐标系”）
ROI_CENTER = np.array([700.0, 1000.0], dtype=np.float32)
ROI_RADIUS = 100.0
MIN_PTS = 500  # ROI 点数太少会不稳，可按需调小/调大

# ------------------- 1) 读入h5ad -------------------
adataTrans1 = sc.read_h5ad("/p1/data/jtian/SA/embyro/output/demo1_adataTrans1.h5ad")
adataMeta1  = sc.read_h5ad("/p1/data/jtian/SA/embyro/output/demo1_adataMeta1.h5ad")

print("Trans1:", type(adataTrans1.X), adataTrans1.X.shape)
print("Meta1 :", type(adataMeta1.X),  adataMeta1.X.shape)
print("n_trans, n_meta:", adataTrans1.n_obs, adataMeta1.n_obs)
print("n_vars trans/meta:", adataTrans1.n_vars, adataMeta1.n_vars)

# ------------------- 2) 取坐标 -------------------
def get_xy(adata):
    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        return adata.obs[["x", "y"]].to_numpy().astype(np.float32)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"]).astype(np.float32)
    raise ValueError("找不到坐标：需要 adata.obs['x','y'] 或 adata.obsm['spatial']")

XY_trans = get_xy(adataTrans1)
XY_meta  = get_xy(adataMeta1)

# 将坐标平移+统一缩放到一个目标范围（保持长宽比例）
def norm_xy_to(XY, target=2000):
    XY = XY.astype(np.float32)
    XY = XY - XY.min(axis=0)
    mx = XY.max(axis=0)
    s = target / (mx.max() + 1e-8)
    XY = XY * s
    return XY

coords_raw = {
    "trans": norm_xy_to(XY_trans, target=2000),
    "meta":  norm_xy_to(XY_meta,  target=2000),
}

print("XY_trans min/max:", coords_raw["trans"].min(0), coords_raw["trans"].max(0))
print("XY_meta  min/max:", coords_raw["meta"].min(0),  coords_raw["meta"].max(0))

# ------------------- 3) 各自降维到同一个 k 维 -------------------
# 空转：稀疏超大矩阵 -> 用 TruncatedSVD（稀疏版 PCA）
svd = TruncatedSVD(n_components=k, random_state=0)
Z_trans = svd.fit_transform(adataTrans1.X)  # (n_trans, k) dense

# 空代：194维 dense -> PCA
pca = PCA(n_components=k, random_state=0)
X_meta = adataMeta1.X
if issparse(X_meta):
    X_meta = X_meta.toarray()
Z_meta = pca.fit_transform(X_meta.astype(np.float32))  # (n_meta, k)

print("Z_trans:", Z_trans.shape, "Z_meta:", Z_meta.shape)

# ------------------- 4) 对 k 维表示再做一次标准化 -------------------
scaler_trans = StandardScaler()
scaler_meta  = StandardScaler()
Z_trans = scaler_trans.fit_transform(Z_trans).astype(np.float32)
Z_meta  = scaler_meta.fit_transform(Z_meta).astype(np.float32)

# ------------------- 5) 构造全量 embed_dict（可复用） -------------------
# 建议用 float32（ROI 小了，内存压力不大；float16 可能不稳）
embed_dict_full = {
    "trans": torch.tensor(Z_trans, dtype=torch.float32),
    "meta":  torch.tensor(Z_meta,  dtype=torch.float32),
}
torch.save(embed_dict_full, os.path.join(output_path, "embed_dict_full.pt"))
samples = np.array(["trans", "meta"], dtype=object)
kmeans_plot_multiple(embed_dict_full, samples, coords_raw, 'kmeans_pca', output_path, k=20, dot_size=10, minibatch=False)
# ------------------- 6) ROI 取子集：同时裁剪 coords_raw + embed_dict -------------------
def circle_mask(XY, center, radius):
    center = center.astype(np.float32)
    d2 = np.sum((XY - center) ** 2, axis=1)
    return d2 <= (radius ** 2)

mask_trans = circle_mask(coords_raw["trans"], ROI_CENTER, ROI_RADIUS)
mask_meta  = circle_mask(coords_raw["meta"],  ROI_CENTER, ROI_RADIUS)

n_trans_roi = int(mask_trans.sum())
n_meta_roi  = int(mask_meta.sum())
print(f"ROI counts - trans: {n_trans_roi}, meta: {n_meta_roi}")

if n_trans_roi < MIN_PTS or n_meta_roi < MIN_PTS:
    print("⚠️ ROI 点数太少，可能导致对齐不稳定。"
          "可尝试增大 ROI_RADIUS 或调低 MIN_PTS。")

# 裁剪坐标
coords_raw_roi = {
    "trans": coords_raw["trans"][mask_trans],
    "meta":  coords_raw["meta"][mask_meta],
}

# 裁剪 embedding（关键：必须与坐标一一对应）
embed_dict_roi = {
    "trans": embed_dict_full["trans"][mask_trans],
    "meta":  embed_dict_full["meta"][mask_meta],
}

# 保存 ROI 索引，方便回写/追踪
idx_trans_roi = np.where(mask_trans)[0]
idx_meta_roi  = np.where(mask_meta)[0]
np.save(os.path.join(output_path, "idx_trans_roi.npy"), idx_trans_roi)
np.save(os.path.join(output_path, "idx_meta_roi.npy"), idx_meta_roi)

print("coords_raw_roi shapes:", coords_raw_roi["trans"].shape, coords_raw_roi["meta"].shape)
print("embed_dict_roi shapes:", embed_dict_roi["trans"].shape, embed_dict_roi["meta"].shape)

# ------------------- 7) CAST.STACK 对齐（只对 ROI 子集） -------------------
graph_list = ["trans", "meta"]

params_dist = CAST.reg_params(
    dataname=graph_list[0],
    gpu=device_gpu,
    diff_step=5,
    iterations=1500,
    dist_penalty1=0.1,
    bleeding=20,  # ROI 小时可以小一些；若对齐不稳可增大到 50~200
    d_list=[3, 2, 1, 1/2, 1/3],
    attention_params=[None, 3, 1, 0],
    dist_penalty2=[0],
    alpha_basis_bs=[500],
    meshsize=[8],
    iterations_bs=[400],
    attention_params_bs=[[None, 3, 1, 0]],
    mesh_weight=[None],
)

params_dist.alpha_basis = torch.tensor(
    [1/1000, 1/1000, 1/50, 5, 5],
    dtype=torch.float32
).reshape(5, 1).to(params_dist.device)

coords_final_roi = CAST.CAST_STACK(
    coords_raw_roi,
    embed_dict_roi,
    output_path,
    graph_list,
    params_dist=params_dist,
    load_affine_ckpt=False,
    save_affine_ckpt=True,
    affine_ckpt_path=None,
)

torch.save(coords_final_roi, os.path.join(output_path, "coords_final_roi.pt"))

print("完成对齐（ROI）。coords_final_roi keys:", coords_final_roi.keys())
print("trans aligned coords shape:", coords_final_roi["trans"].shape)
print("meta  coords shape:", coords_final_roi["meta"].shape)
print("trans aligned head:\n", coords_final_roi["trans"][:5])
print("meta head:\n", coords_final_roi["meta"][:5])
