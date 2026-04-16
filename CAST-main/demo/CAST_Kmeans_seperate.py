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


output_path = Path("/p2/zulab/jtian/data/SA/06_calculateConcentration/output_CAST_Kmeans_seperate/")
output_path.mkdir(parents=True, exist_ok=True)
replicates = ["ctrl1", "ctrl2", "ctrl3", "dn1", "dn2", "dn3"]
slice_list = [0, 15, 30, 45, 60]
adata = ad.read_h5ad("/p2/zulab/jtian/data/SA/06_calculateConcentration/output_CAST_Leiden1/adata30.h5ad")
print("norm_1e4_log1p:")
print(adata.layers['norm_1e4_log1p'][:5])
print(np.min(adata.layers['norm_1e4_log1p']), np.max(adata.layers['norm_1e4_log1p']))
adata.layers['norm_1e4_log1p_scale'] = adata.layers["norm_1e4_log1p"].copy()
sc.pp.scale(adata.layers['norm_1e4_log1p_scale'], zero_center=False)
print("norm_1e4_log1p_scale:")
print(adata.layers['norm_1e4_log1p_scale'][:5])
print(np.min(adata.layers['norm_1e4_log1p_scale']), np.max(adata.layers['norm_1e4_log1p_scale']))
OUTPUT_H5AD = f'{output_path}/adata30.h5ad'
adata.write_h5ad(OUTPUT_H5AD)

adata.obs["replicate"] = adata.obs["sample"].astype(str).str.split("_").str[0]
group_names = adata.obs["replicate"].unique().tolist()
adata_dict = {s: adata[adata.obs["replicate"] == s].copy() for s in group_names}

for s, adata_sub in adata_dict.items():
    out_file = output_path / f"adata_{s}.h5ad"
    adata_sub.write_h5ad(out_file)
    print(f"已保存: {out_file}")

###########################CAST.MARK##########################################
cast_output_root = Path("/p2/zulab/jtian/data/SA/06_calculateConcentration/output_CAST_Kmeans_seperate/CAST_MARK_by_sample/")
cast_output_root.mkdir(parents=True, exist_ok=True)
# group_names = ['ctrl1', 'ctrl2', 'ctrl3', 'dn1', 'dn2', 'dn3']
for s in group_names:
    adata_use = ad.read_h5ad(output_path / f"adata_{s}.h5ad")
    samples = np.unique(adata_use.obs["sample"])
    coords_raw = {sample_t: np.array(adata_use.obs[["x", "y"]])[adata_use.obs["sample"] == sample_t]for sample_t in samples}
    exp_dict = {sample_t: adata_use[adata_use.obs["sample"] == sample_t].layers["norm_1e4_log1p_scale"]for sample_t in samples}

    sample_out = cast_output_root / f'{s}'
    sample_out.mkdir(parents=True, exist_ok=True)
    args = Args(
        dataname='task1',
        gpu=0,
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

    embed_dict = CAST_MARK(coords_raw, exp_dict, sample_out, args=args)
    GLOBAL_K=20
    _, global_label_dict = kmeans_plot_multiple(
        embed_dict,
        samples,
        coords_raw,
        'demo1',
        sample_out,
        k=GLOBAL_K,
        dot_size=10,
        minibatch=False,
        return_label_dict=True,
        save_label_dict=True)

































































