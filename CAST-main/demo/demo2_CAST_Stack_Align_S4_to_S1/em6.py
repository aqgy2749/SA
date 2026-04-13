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
from CAST import CAST_STACK
from CAST.CAST_Stack import reg_params
from CAST.models.model_GCNII import Args
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
output_path='/p2/zulab/jtian/data/SA/embyro/output_em6'
os.makedirs(output_path,exist_ok=True)

adataMeta1_roi = sc.read_h5ad("/p2/zulab/jtian/data/SA/embyro/output_em5/adataMeta1_roi.h5ad")
adataMeta2_roi = sc.read_h5ad("/p2/zulab/jtian/data/SA/embyro/output_em5/adataMeta2_roi.h5ad")
adataTrans1_roi = sc.read_h5ad("/p2/zulab/jtian/data/SA/embyro/output_em5/adataTrans1_roi.h5ad")
adataTrans2_roi = sc.read_h5ad("/p2/zulab/jtian/data/SA/embyro/output_em5/adataTrans2_roi.h5ad")
GPU_ID = 0 if torch.cuda.is_available() else -1
def get_xy(adata):
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

def to_float32_array(X):
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.toarray().astype(np.float32)
    except Exception:
        pass
    return np.asarray(X, dtype=np.float32)

def get_hvg_mask(adata, fallback_top=2000):
    """
    优先使用 adata.var['highly_variable']。
    如果不存在或数量不够/为0，则退化为取前 fallback_top 个基因（按当前 var 顺序）。
    """
    if "highly_variable" in adata.var.columns:
        hv = adata.var["highly_variable"].to_numpy()
        if hv.dtype != bool:
            hv = hv.astype(bool)
        cnt = int(hv.sum())
        if cnt > 0:
            return hv
    # fallback：直接取前 fallback_top
    hv = np.zeros(adata.n_vars, dtype=bool)
    hv[:min(fallback_top, adata.n_vars)] = True
    return hv

#trans1 meta1

# --- 取 trans HVG (约2000) ---
hvg = get_hvg_mask(adataTrans1_roi, fallback_top=2000)
X_trans = to_float32_array(adataTrans1_roi[:, hvg].X)   # (n_trans_roi, ~2000)
X_meta  = to_float32_array(adataMeta1_roi.X) 
exp_dict = {
    "trans1": X_trans.astype(np.float32),
    "meta1":  X_meta.astype(np.float32),
}
coords_raw = {
    "trans1": np.floor(norm_xy_to(get_xy(adataTrans1_roi), target=2000)),
    "meta1":  np.floor(norm_xy_to(get_xy(adataMeta1_roi),  target=2000)),
}
for k,v in coords_raw.items():
    print(k, v.shape, "min", v.min(axis=0), "max", v.max(axis=0))

args = Args(
    dataname='e1',
    gpu=GPU_ID,
    epochs=2000,
    lr1=1e-3,
    wd1=0,
    lambd=1e-3,
    n_layers=9,
    der=0.5,
    dfr=0.3,
    use_encoder=True,
    encoder_dim=100,
)
embed_dict = CAST_MARK(coords_raw,exp_dict,output_path,args=args)
# embed_dict = torch.load(f'{output_path}/demo_embed_dict.pt',map_location='cpu')

samples=['trans1','meta1']
cell_label = kmeans_plot_multiple(embed_dict,samples,coords_raw,'demo1',output_path,k=20,dot_size = 10,minibatch=False)
torch.save(cell_label, os.path.join(output_path, f"{samples[0]}_{samples[1]}_cell_label_dict_k20.pt"))
graph_list = ['trans1','meta1']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=3000,#500
                            dist_penalty1=0.1,#0
                            bleeding=500,#500
                            d_list = [3,2,1,1/2,1/3],
                            attention_params = [None,3,1,0],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [0],
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list,params_dist=params_dist,load_affine_ckpt=False,save_affine_ckpt=True,affine_ckpt_path=None)
# coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list,params_dist=params_dist,load_affine_ckpt=True,save_affine_ckpt=False,affine_ckpt_path=None,strict_affine_ckpt_match=True)


#trans2 meta2

# --- 取 trans HVG (约2000) ---
hvg = get_hvg_mask(adataTrans2_roi, fallback_top=2000)
X_trans = to_float32_array(adataTrans2_roi[:, hvg].X)   # (n_trans_roi, ~2000)
X_meta  = to_float32_array(adataMeta2_roi.X) 
exp_dict = {
    "trans2": X_trans.astype(np.float32),
    "meta2":  X_meta.astype(np.float32),
}
coords_raw = {
    "trans2": np.floor(norm_xy_to(get_xy(adataTrans2_roi), target=2000)),
    "meta2":  np.floor(norm_xy_to(get_xy(adataMeta2_roi),  target=2000)),
}
for k,v in coords_raw.items():
    print(k, v.shape, "min", v.min(axis=0), "max", v.max(axis=0))

args = Args(
    dataname='e2',
    gpu=GPU_ID,
    epochs=2000,
    lr1=1e-3,
    wd1=0,
    lambd=1e-3,
    n_layers=9,
    der=0.5,
    dfr=0.3,
    use_encoder=True,
    encoder_dim=100,
)
embed_dict = CAST_MARK(coords_raw,exp_dict,output_path,args=args)
# embed_dict = torch.load(f'{output_path}/demo_embed_dict.pt',map_location='cpu')

samples=['trans2','meta2']
cell_label = kmeans_plot_multiple(embed_dict,samples,coords_raw,'demo1',output_path,k=20,dot_size = 10,minibatch=False)
torch.save(cell_label, os.path.join(output_path, f"{samples[0]}_{samples[1]}_cell_label_dict_k20.pt"))
graph_list = ['trans2','meta2']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=3000,#500
                            dist_penalty1=0.1,#0
                            bleeding=500,#500
                            d_list = [3,2,1,1/2,1/3],
                            attention_params = [None,3,1,0],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [0],
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list,params_dist=params_dist,load_affine_ckpt=False,save_affine_ckpt=True,affine_ckpt_path=None)
# coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list,params_dist=params_dist,load_affine_ckpt=True,save_affine_ckpt=False,affine_ckpt_path=None,strict_affine_ckpt_match=True)
