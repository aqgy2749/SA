import os
import torch
import numpy as np
import anndata as ad
import scanpy as sc
import CAST
import warnings
warnings.filterwarnings("ignore")
work_dir = '/p1/data/jtian/SA/05_CAST' #### input the demo path
output_path = f'{work_dir}/output_log+scale+shift_repeat'
os.makedirs(output_path,exist_ok = True)
adata = ad.read_h5ad(f'{output_path}/demo1_preprocessed.h5ad')
adata.layers['norm_1e4'] = adata.X
#adata.layers['norm_1e4'] = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X'].toarray() # we use normalized counts for each cell as input gene expression
######我们的数据不做归一化，我们数据也不是稀疏矩阵，不需要toarray转数组
samples = np.unique(adata.obs['sample']) # used samples in adata
coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs['sample'] == sample_t] for sample_t in samples}
for s in samples:
    anchor = coords_raw[s][0].copy()   
    coords_raw[s] = coords_raw[s] - anchor
exp_dict = {sample_t: adata[adata.obs['sample'] == sample_t].layers['norm_1e4'] for sample_t in samples}
embed_dict = torch.load(f'{output_path}/demo_embed_dict.pt',map_location='cpu')
graph_list = ['15','0'] # [query_sample, reference_sample]
query_sample = graph_list[0]
params_dist = CAST.reg_params(
    dataname=query_sample,
    gpu=0 if torch.cuda.is_available() else -1,
    diff_step=1,  # 减小步长，帮助更平稳的收敛
    iterations=3000, #3000,  # 增加迭代次数，以便更多的训练
    dist_penalty1=0.1,  # 加入少量正则化，避免过拟合
    bleeding=200,  # 减小平移的幅度
    d_list=[3, 2, 1, 1/2, 1/3],
    attention_params=[None, 3, 1, 0],

    # FFD参数随便填，但关键是 iterations_bs[0] = 0
    dist_penalty2=[0],
    alpha_basis_bs=[500],
    meshsize=[8],
    iterations_bs=[400],  # 只跑 Affine，不执行 FFD
    attention_params_bs=[[None, 3, 1, 0]],
    mesh_weight=[None]
)

params_dist.alpha_basis = torch.Tensor([1/2000, 1/2000, 1/100, 3, 3]).reshape(5, 1).to(params_dist.device)

# 运行训练
coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list, params_dist)
