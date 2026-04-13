from pathlib import Path
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
import anndata as ad

base_input = Path("/p2/zulab/jtian/data/SA/05_CAST/input")
intensity_root = base_input / "LipidsIntensity"
coord_root = base_input / "LipidsSpotsIndex"

output_path = Path("/p2/zulab/jtian/data/SA/06_calculateConcentration/output_CAST_Leiden1/")
output_path.mkdir(parents=True, exist_ok=True)
replicates = ["ctrl1", "ctrl2", "ctrl3", "dn1", "dn2", "dn3"]
slice_list = [0, 15, 30, 45, 60]
# adata_list = []
# for rep in replicates:
#     for s in slice_list:
#         sample_name = f"{rep}_{s}"

#         intensity_file = intensity_root / rep / f"lipid{s}.csv"
#         coord_file = coord_root / rep / f"lipids{s}RegionSpots.csv"
#         intensity_df = pd.read_csv(intensity_file, sep=";", header=0, index_col=0)
#         print(f"{sample_name} 原始 intensity 形状(基因 × 细胞): {intensity_df.shape}")
#         expr_df = intensity_df.T.copy()
#         print(f"{sample_name} 转置后 expr 形状(细胞 × 基因): {expr_df.shape}")

#         coord_df = pd.read_csv(coord_file, sep=";", header=0)
#         print(f"{sample_name} 原始坐标形状: {coord_df.shape}")
#         x = coord_df['x'].to_numpy()
#         y = coord_df['y'].to_numpy()

#         # -------------------------------------------------
#         # 3.4 构建唯一细胞名
#         # 避免 30 个切片合并后 cell barcode 重复
#         # -------------------------------------------------
#         pixel_ids = [f"{sample_name}_pixel{i}" for i in range(1, expr_df.shape[0] + 1)]
#         meta_ids = expr_df.columns.astype(str)
#         expr_df.index = pixel_ids
#         expr_df.columns = meta_ids

#         obs = pd.DataFrame(index=pixel_ids)
#         obs["sample"] = sample_name
#         obs["x"] = x
#         obs["y"] = y

#         var = pd.DataFrame(index=meta_ids)

#         # -------------------------------------------------
#         # 3.6 构建单切片 AnnData
#         # -------------------------------------------------
#         adata_one = ad.AnnData(
#             X=expr_df.to_numpy(dtype=np.float32),
#             obs=obs,
#             var=var
#         )

#         # 空间坐标放进 obsm['spatial']
#         adata_one.obsm["spatial"] = np.column_stack([x, y]).astype(np.float32)

#         print(f"{sample_name} AnnData 形状: {adata_one.shape}")
#         print(f"{sample_name} X 前5行前5列:")
#         print(adata_one.X[:5, :5])

#         adata_list.append(adata_one)
# print(all(adata_list[0].var_names.equals(x.var_names) for x in adata_list))
# adata = ad.concat(adata_list,axis=0,)
# print(f"合并后 adata 形状: {adata.shape}")
# print("各 sample 细胞数:")
# print(adata.obs["sample"].value_counts().sort_index())
# adata.layers['raw']=adata.X.copy()
# sc.pp.log1p(adata)
# sc.pp.scale(adata, zero_center=False)
# print("log变换+scale")
# print(adata.X[:5])
# print(np.min(adata.X), np.max(adata.X))
# OUTPUT_H5AD = f'{output_path}/adata30.h5ad'
# adata.write_h5ad(OUTPUT_H5AD)
adata = ad.read_h5ad("/p2/zulab/jtian/data/SA/06_calculateConcentration/output_CAST_Leiden/adata30.h5ad")
print("log变换+scale:")
print(adata.X[:5])
print(np.min(adata.X), np.max(adata.X))
adata.X=adata.layers['raw'].copy()
print("raw:")
print(adata.X[:5])
print(np.min(adata.X), np.max(adata.X))
adata.layers['norm_1e4'] = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X']
print("norm_1e4:")
print(adata.layers['norm_1e4'][:5])
print(np.min(adata.layers['norm_1e4']), np.max(adata.layers['norm_1e4']))
adata.layers["norm_1e4_log1p"] = adata.layers["norm_1e4"].copy()
sc.pp.log1p(adata.layers["norm_1e4_log1p"])
print("norm_1e4_log1p:")
print(adata.layers['norm_1e4_log1p'][:5])
print(np.min(adata.layers['norm_1e4_log1p']), np.max(adata.layers['norm_1e4_log1p']))
OUTPUT_H5AD = f'{output_path}/adata30.h5ad'
adata.write_h5ad(OUTPUT_H5AD)

###########################CAST.MARK##########################################
adata = ad.read_h5ad(f'{output_path}/adata30.h5ad')                #########################
samples = np.unique(adata.obs['sample'])
coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs['sample'] == sample_t] for sample_t in samples}
exp_dict = {sample_t: adata[adata.obs['sample'] == sample_t].layers["norm_1e4_log1p"] for sample_t in samples}
args = Args(
            dataname='task1', # name of the dataset, used to save the log file
            gpu = 0, # gpu id, set to zero for single-GPU nodes
            epochs=2000,#400   # number of epochs for training
            lr1= 1e-3, # learning rate  
            wd1= 0, # weight decay
            lambd= 1e-3, # lambda in the loss function, refer to online methods
            n_layers=9, # number of GCNII layers, more layers mean a deeper model, larger reception field, at a cost of VRAM usage and computation time
            der=0.5, # edge dropout rate in CCA-SSG
            dfr=0.3, # feature dropout rate in CCA-SSG
            use_encoder=True, # perform a single-layer dimension reduction before the GNNs, helps save VRAM and computation time if the gene panel is large
            encoder_dim=100, #512  # encoder dimension, ignore if `use_encoder` set to `False`
        )
embed_dict = CAST_MARK(coords_raw,exp_dict,output_path,args=args)


































































