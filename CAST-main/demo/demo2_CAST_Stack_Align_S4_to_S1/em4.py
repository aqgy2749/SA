import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings

import numpy as np
import pandas as pd
import torch
import dgl

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

from scipy.sparse import issparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

import CAST
from CAST import CAST_MARK, CAST_STACK
from CAST.visualize import kmeans_plot_multiple
from CAST.CAST_Stack import reg_params
from CAST.models.model_GCNII import Args



# ===================== 路径配置（按你的 embyroPreprocess 输出） =====================
work_dir = "/p2/zulab/jtian/data/SA/embyro"
output_path = f"{work_dir}/output_em4"
os.makedirs(output_path, exist_ok=True)

TRANS1_H5AD = f"/p1/data/jtian/SA/embyro/output/demo1_adataTrans1.h5ad"
META1_H5AD  = f"/p1/data/jtian/SA/embyro/output/demo1_adataMeta1.h5ad"

# GPU_ID = 0 if torch.cuda.is_available() else -1
GPU_ID = -1

PCA_N_COMPONENTS = 100  # 维度要一致（对应你以前 encoder_dim=100）

# ===================== 读数据 =====================
adataTrans1 = sc.read_h5ad(TRANS1_H5AD)
adataMeta1  = sc.read_h5ad(META1_H5AD)
trans_name = str(np.unique(adataTrans1.obs["sample"])[0])
meta_name  = str(np.unique(adataMeta1.obs["sample"])[0])
print("Trans sample =", trans_name, "| Meta sample =", meta_name)

# ===================== coords_raw =====================
def get_xy(adata):
    # 优先 obs[x,y]，否则 obsm["spatial"]
    if ("x" in adata.obs.columns) and ("y" in adata.obs.columns):
        return np.asarray(adata.obs[["x", "y"]], dtype=np.float32)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=np.float32)
    raise ValueError("找不到坐标：需要 adata.obs['x','y'] 或 adata.obsm['spatial']")

def norm_xy_to(XY, target=2000):
    XY = XY.astype(np.float32)
    XY = XY - XY.min(axis=0)
    mx = XY.max(axis=0)
    s = target / (mx.max() + 1e-8)
    return XY * s

coords_raw = {
    trans_name: norm_xy_to(get_xy(adataTrans1), target=2000),
    meta_name:  norm_xy_to(get_xy(adataMeta1),  target=2000)
}
print("coords shapes:", {k: v.shape for k, v in coords_raw.items()})

# ===================== Trans1：用 HVG 子集做降维 =====================
hvg_mask = adataTrans1.var["highly_variable"].values
n_hvg = int(np.sum(hvg_mask))
print("Trans1 HVG count =", n_hvg)
X_trans = adataTrans1[:, hvg_mask].X  
trans_reducer = TruncatedSVD(n_components=PCA_N_COMPONENTS, random_state=0)
Z_trans = trans_reducer.fit_transform(X_trans)
Z_trans = StandardScaler(with_mean=True, with_std=True).fit_transform(Z_trans)
# ===================== Meta1：正常用全部特征做 PCA =====================
X_meta = adataMeta1.X
X_meta = np.asarray(X_meta, dtype=np.float32)
# X_meta = StandardScaler(with_mean=True, with_std=True).fit_transform(X_meta)
meta_reducer = PCA(n_components=PCA_N_COMPONENTS, random_state=0)
Z_meta = meta_reducer.fit_transform(X_meta)

# ===================== 组装 embed_dict（同一个 dict，两个 key） =====================
embed_dict = {
    trans_name: torch.tensor(Z_trans, dtype=torch.float32),
    meta_name:  torch.tensor(Z_meta, dtype=torch.float32),
}
print("Embed shapes:", trans_name, embed_dict[trans_name].shape, "|", meta_name, embed_dict[meta_name].shape)
samples = [trans_name, meta_name]
# 保存（你要的产物：embed_dict.pt）
embed_path = os.path.join(output_path, "demo_embed_dict.pt")
torch.save(embed_dict, embed_path)
print("Saved embed_dict to:", embed_path)
cell_label=kmeans_plot_multiple(embed_dict,samples,coords_raw,'demo1',output_path,k=20,dot_size = 10,minibatch=False)
torch.save(cell_label, f'{output_path}/cell_label.pt')
# ===================== CAST.STACK：只跑 trans1 -> meta1 一次 =====================
graph_list = [trans_name, meta_name]

params_dist = CAST.reg_params(
    dataname=graph_list[0],
    gpu=GPU_ID,
    diff_step=5,

    # Affine
    iterations=10,
    dist_penalty1=0.1,
    bleeding=2000,
    d_list=[3, 2, 1, 1/2, 1/3],
    attention_params=[None, 3, 1, 0],

    # FFD（
    dist_penalty2=[0],
    alpha_basis_bs=[500],
    meshsize=[8],
    iterations_bs=[0],
    attention_params_bs=[[None, 3, 1, 0]],
    mesh_weight=[None],
)
# params_dist.alpha_basis = torch.Tensor([1/1000, 1/1000, 1/50, 5, 5]).reshape(5, 1).to(params_dist.device)
params_dist.alpha_basis = torch.tensor(
    [1/1000, 1/1000, 1/50, 5, 5],
    dtype=torch.float32,
    device="cpu"
).reshape(5, 1)

coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list,params_dist=params_dist,load_affine_ckpt=False,save_affine_ckpt=True,affine_ckpt_path=None)
# coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list,params_dist=params_dist,load_affine_ckpt=True,save_affine_ckpt=False,affine_ckpt_path=None,strict_affine_ckpt_match=True)


# 可选：把 coords_final 也存一下（方便后面 projection/可视化）
torch.save(coords_final, os.path.join(output_path, "coords_final.pt"))
print("Saved coords_final to:", os.path.join(output_path, "coords_final.pt"))
