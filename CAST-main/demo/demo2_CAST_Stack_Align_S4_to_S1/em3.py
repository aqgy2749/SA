import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import numpy as np
import anndata as ad

import CAST
from CAST import CAST_MARK
from CAST.models.model_GCNII import Args
from CAST.visualize import kmeans_plot_multiple

# ===================== 0) 路径配置 =====================
work_dir = "/p1/data/jtian/SA/embyro"
out_root = f"{work_dir}/output_em3"
os.makedirs(out_root, exist_ok=True)

TRANS1_H5AD = f"{work_dir}/output/demo1_adataTrans1.h5ad"
TRANS2_H5AD = f"{work_dir}/output/demo1_adataTrans2.h5ad"
META1_H5AD  = f"{work_dir}/output/demo1_adataMeta1.h5ad"
META2_H5AD  = f"{work_dir}/output/demo1_adataMeta2.h5ad"

GPU_ID = 0 if torch.cuda.is_available() else -1

# ===================== 1) 工具函数 =====================
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

# ===================== 2) 读取数据 =====================
adataTrans1 = ad.read_h5ad(TRANS1_H5AD)
adataTrans2 = ad.read_h5ad(TRANS2_H5AD)
adataMeta1  = ad.read_h5ad(META1_H5AD)
adataMeta2  = ad.read_h5ad(META2_H5AD)

print("Trans1:", adataTrans1.shape, type(adataTrans1.X))
print("Trans2:", adataTrans2.shape, type(adataTrans2.X))
print("Meta1 :", adataMeta1.shape,  type(adataMeta1.X))
print("Meta2 :", adataMeta2.shape,  type(adataMeta2.X))

# # ===================== 3) 定义“成对整合”的函数 =====================
# def run_pair(trans_adata, meta_adata, pair_name, out_root, gpu_id):
#     out_dir = os.path.join(out_root, pair_name)
#     os.makedirs(out_dir, exist_ok=True)

#     # --- 取 trans HVG (约2000) ---
#     hvg = get_hvg_mask(trans_adata, fallback_top=2000)
#     X_trans = to_float32_array(trans_adata[:, hvg].X)   # (n_trans, 2000)
#     X_meta  = to_float32_array(meta_adata.X)            # (n_meta, 194)

#     print(f"\n[{pair_name}] X_trans: {X_trans.shape} (HVG count = {int(hvg.sum())}), X_meta: {X_meta.shape}")

#     # ✅ 不再 pad：多 encoder 会自己处理不同 in_dim
#     exp_dict = {
#         "trans": X_trans.astype(np.float32),
#         "meta":  X_meta.astype(np.float32),
#     }
#     coords_raw = {
#         "trans": norm_xy_to(get_xy(trans_adata), target=2000),
#         "meta":  norm_xy_to(get_xy(meta_adata),  target=2000),
#     }

#     # 行数检查
#     assert coords_raw["trans"].shape[0] == exp_dict["trans"].shape[0]
#     assert coords_raw["meta"].shape[0]  == exp_dict["meta"].shape[0]

#     # MARK 参数（encoder_dim 是“统一后的维度”，trans/meta 都会投到这里）
#     args = Args(
#         dataname=pair_name,
#         gpu=gpu_id,
#         epochs=600,
#         lr1=1e-3,
#         wd1=0,
#         lambd=1e-3,
#         n_layers=9,
#         der=0.5,
#         dfr=0.3,
#         use_encoder=True,
#         encoder_dim=100,
#     )

#     # 跑 CAST_MARK（你库里 CAST_MARK 已经改成 multi-encoder 版本）
#     embed_dict = CAST_MARK(coords_raw, exp_dict, out_dir, args=args)


#     # kmeans 可视化
#     samples = ["trans", "meta"]
#     for k in [10, 15, 20]:
#         labels = kmeans_plot_multiple(
#             embed_dict,
#             samples,
#             coords_raw,
#             f"{pair_name}_kmeans_k{k}",
#             out_dir,
#             k=k,
#             dot_size=10,
#             minibatch=False
#         )
#         torch.save(labels, os.path.join(out_dir, f"cell_label_dict_k{k}.pt"))
#         print(f"[{pair_name}] saved labels: cell_label_dict_k{k}.pt")

#     return embed_dict


# # ===================== 4) 分别整合 trans1+meta1, trans2+meta2 =====================
# embed_pair1 = run_pair(adataTrans1, adataMeta1, "trans1_meta1", out_root, GPU_ID)
# embed_pair2 = run_pair(adataTrans2, adataMeta2, "trans2_meta2", out_root, GPU_ID)

# print("\nDone. Outputs in:", out_root)
##############################CAST.STACK##########################################
coords_raw = {
        "trans": norm_xy_to(get_xy(adataTrans1), target=2000),
        "meta":  norm_xy_to(get_xy(adataMeta1),  target=2000),
    }
print("trans1 xy min/max:", coords_raw["trans"].min(0), coords_raw["trans"].max(0))
print("meta1  xy min/max:", coords_raw["meta"].min(0),  coords_raw["meta"].max(0))
embed_dict = torch.load("/p1/data/jtian/SA/embyro/output_em3/trans1_meta1/demo_embed_dict.pt",map_location='cpu')
graph_list = ['trans','meta']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            # gpu = 0 if torch.cuda.is_available() else -1,
                            gpu = -1,
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
# params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
params_dist.alpha_basis = torch.tensor(
    [1/1000, 1/1000, 1/50, 5, 5],
    dtype=torch.float32,
    device="cpu"
).reshape(5, 1)

output_path="/p1/data/jtian/SA/embyro/output_em3/trans1_meta1"
print("CUDA available:", torch.cuda.is_available())
print("params_dist device:", params_dist.device)
coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list,params_dist=params_dist,load_affine_ckpt=False,save_affine_ckpt=True,affine_ckpt_path=None)
# coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list,params_dist=params_dist,load_affine_ckpt=True,save_affine_ckpt=False,affine_ckpt_path=None,strict_affine_ckpt_match=True)
