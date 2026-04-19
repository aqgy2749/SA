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


warnings.filterwarnings("ignore")
output_path = '/p2/zulab/jtian/data/SA/06_calculateConcentration/output_cast18'      ############
os.makedirs(output_path, exist_ok=True)

adata_use = ad.read_h5ad("/p2/zulab/jtian/data/SA/06_calculateConcentration/output_CAST_Kmeans_seperate/adata_ctrl3.h5ad")
samples = np.unique(adata_use.obs["sample"])
coords_raw = {sample_t: np.array(adata_use.obs[["x", "y"]])[adata_use.obs["sample"] == sample_t]for sample_t in samples}
exp_dict = {sample_t: adata_use[adata_use.obs["sample"] == sample_t].layers["norm_1e4_log1p_scale"]for sample_t in samples}
embed_dict = torch.load("/p2/zulab/jtian/data/SA/06_calculateConcentration/output_CAST_Kmeans_seperate/CAST_MARK_by_sample/ctrl3/demo_embed_dict.pt",map_location='cpu')
GLOBAL_K = 20
label_path="/p2/zulab/jtian/data/SA/06_calculateConcentration/output_CAST_Kmeans_seperate/CAST_MARK_by_sample/ctrl3/demo1_kmeans_labels_k20.npz"
npz_file = np.load(label_path, allow_pickle=True)
global_label_dict = {key: npz_file[key] for key in npz_file.files}
###########################CAST.STACK##########################################
graph_list = ['ctrl3_15','ctrl3_0']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=3000,#500
                            dist_penalty1=0.1,#0
                            bleeding=200,#500
                            d_list = [3,2,1,1/2,1/3],
                            attention_params = [None,3,1,0],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [400],
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
coords_final = CAST_STACK(
    coords_raw,
    embed_dict,
    output_path,
    graph_list,
    params_dist=params_dist,
    save_affine_ckpt=True,
    load_affine_ckpt=False,
    global_label_dict=global_label_dict,
    global_k=GLOBAL_K
)

graph_list = ['ctrl3_30','ctrl3_0']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=2000,#500
                            dist_penalty1=0.1,#0
                            bleeding=200,#500
                            d_list = [3,2,1,1/2,1/3],
                            attention_params = [None,3,1,0],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [400],
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
coords_final = CAST_STACK(
    coords_raw,
    embed_dict,
    output_path,
    graph_list,
    params_dist=params_dist,
    save_affine_ckpt=True,
    load_affine_ckpt=False,
    global_label_dict=global_label_dict,
    global_k=GLOBAL_K
)

graph_list = ['ctrl3_45','ctrl3_0']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=5000,#500
                            dist_penalty1=0.1,#0
                            bleeding=200,#500
                            d_list = [3,2,1,1/2,1/3],
                            attention_params = [None,3,1,0],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [400],
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
coords_final = CAST_STACK(
    coords_raw,
    embed_dict,
    output_path,
    graph_list,
    params_dist=params_dist,
    save_affine_ckpt=True,
    load_affine_ckpt=False,
    global_label_dict=global_label_dict,
    global_k=GLOBAL_K
)

graph_list = ['ctrl3_60','ctrl3_0']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=5000,#500
                            dist_penalty1=0.1,#0
                            bleeding=200,#500
                            d_list = [3,2,1,1/2,1/3],
                            attention_params = [None,3,1,0],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [400],
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
coords_final = CAST_STACK(
    coords_raw,
    embed_dict,
    output_path,
    graph_list,
    params_dist=params_dist,
    save_affine_ckpt=True,
    load_affine_ckpt=False,
    global_label_dict=global_label_dict,
    global_k=GLOBAL_K
)