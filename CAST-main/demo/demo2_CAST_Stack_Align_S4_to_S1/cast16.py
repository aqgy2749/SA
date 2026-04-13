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
work_dir = '/p2/zulab/jtian/data/SA/05_CAST'
output_path = f'{work_dir}/output_cast16'      ############
os.makedirs(output_path, exist_ok=True)



###########################CAST.MARK##########################################
adata = ad.read_h5ad("/p2/zulab/jtian/data/SA/05_CAST/output_cast15/demo1_cast15.h5ad")                #########################
adata.layers['exp'] = adata.X 
samples = np.unique(adata.obs['sample'])
coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs['sample'] == sample_t] for sample_t in samples}
for s in samples:
    anchor = coords_raw[s][0].copy()   
    coords_raw[s] = coords_raw[s] - anchor
for s in ['15', '30', '45', '60']:
    coords_raw[s] = -coords_raw[s]
    # center = coords_raw[s].mean(axis=0, keepdims=True)
    # coords_raw[s] = 2 * center - coords_raw[s]
exp_dict = {sample_t: adata[adata.obs['sample'] == sample_t].layers['exp'] for sample_t in samples}
args = Args(
            dataname='task1', # name of the dataset, used to save the log file
            gpu = 0, # gpu id, set to zero for single-GPU nodes
            epochs=400,#400   # number of epochs for training
            lr1= 1e-3, # learning rate  
            wd1= 0, # weight decay
            lambd= 1e-3, # lambda in the loss function, refer to online methods
            n_layers=9, # number of GCNII layers, more layers mean a deeper model, larger reception field, at a cost of VRAM usage and computation time
            der=0.5, # edge dropout rate in CCA-SSG
            dfr=0.3, # feature dropout rate in CCA-SSG
            use_encoder=True, # perform a single-layer dimension reduction before the GNNs, helps save VRAM and computation time if the gene panel is large
            encoder_dim=512, #512  # encoder dimension, ignore if `use_encoder` set to `False`
        )
embed_dict = CAST_MARK(coords_raw,exp_dict,output_path,args=args)
kmeans_plot_multiple(embed_dict,samples,coords_raw,'demo1',output_path,k=20,dot_size = 10,minibatch=False)
# embed_dict = torch.load("/p2/zulab/jtian/data/SA/05_CAST/output_cast15/demo_embed_dict.pt",map_location='cpu')
###########################CAST.STACK##########################################
graph_list = ['15','0']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=2000,#500
                            dist_penalty1=0,#0.1
                            bleeding=500,#200
                            d_list = [1,0.9,0.8,1.1,1.2],
                            attention_params = [None,3,1,0],
                            translation_params = [0.4, 0.4, 9],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [400],#400
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,20,5,5]).reshape(5,1).to(params_dist.device)#[1/1000,1/1000,1/50,5,5]
# coords_final = CAST.CAST_STACK(coords_raw,embed_dict,output_path,graph_list,params_dist)
coords_final = CAST_STACK(
    coords_raw,
    embed_dict,
    output_path,
    graph_list,
    params_dist=params_dist,
    save_affine_ckpt=True,
    load_affine_ckpt=False
)

graph_list = ['30','0']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=2000,#500
                            dist_penalty1=0,#0.1
                            bleeding=500,#200
                            d_list = [1,0.9,0.8,1.1,1.2],
                            attention_params = [None,3,1,0],
                            translation_params = [0.4, 0.4, 9],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [400],#400
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,20,5,5]).reshape(5,1).to(params_dist.device)#[1/1000,1/1000,1/50,5,5]
# coords_final = CAST.CAST_STACK(coords_raw,embed_dict,output_path,graph_list,params_dist)
coords_final = CAST_STACK(
    coords_raw,
    embed_dict,
    output_path,
    graph_list,
    params_dist=params_dist,
    save_affine_ckpt=True,
    load_affine_ckpt=False
)

graph_list = ['45','0']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=2000,#500
                            dist_penalty1=0,#0.1
                            bleeding=500,#200
                            d_list = [1,0.9,0.8,1.1,1.2],
                            attention_params = [None,3,1,0],
                            translation_params = [0.4, 0.4, 9],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [400],#400
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,20,5,5]).reshape(5,1).to(params_dist.device)#[1/1000,1/1000,1/50,5,5]
# coords_final = CAST.CAST_STACK(coords_raw,embed_dict,output_path,graph_list,params_dist)
coords_final = CAST_STACK(
    coords_raw,
    embed_dict,
    output_path,
    graph_list,
    params_dist=params_dist,
    save_affine_ckpt=True,
    load_affine_ckpt=False
)
graph_list = ['60','0']
params_dist = CAST.reg_params(dataname = graph_list[0], 
                            gpu = 0 if torch.cuda.is_available() else -1,
                            diff_step = 5,
                            #### Affine parameters
                            iterations=2000,#500
                            dist_penalty1=0,#0.1
                            bleeding=500,#200
                            d_list = [1,0.9,0.8,1.1,1.2],
                            attention_params = [None,3,1,0],
                            translation_params = [0.4, 0.4, 9],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [500],
                            meshsize = [8],
                            iterations_bs = [400],#400
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,20,5,5]).reshape(5,1).to(params_dist.device)#[1/1000,1/1000,1/50,5,5]
# coords_final = CAST.CAST_STACK(coords_raw,embed_dict,output_path,graph_list,params_dist)
coords_final = CAST_STACK(
    coords_raw,
    embed_dict,
    output_path,
    graph_list,
    params_dist=params_dist,
    save_affine_ckpt=True,
    load_affine_ckpt=False
)


