import os
import torch
import numpy as np
import anndata as ad
import scanpy as sc
import CAST
import warnings
warnings.filterwarnings("ignore")
work_dir = '/p1/data/jtian/SA/05_CAST' #### input the demo path
output_path = f'{work_dir}/output_5'
os.makedirs(output_path,exist_ok = True)
adata = ad.read_h5ad(f'{output_path}/demo1.h5ad')
adata.layers['norm_1e4'] = adata.X
#adata.layers['norm_1e4'] = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X'].toarray() # we use normalized counts for each cell as input gene expression
######我们的数据不做归一化，我们数据也不是稀疏矩阵，不需要toarray转数组
samples = np.unique(adata.obs['sample']) # used samples in adata
coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs['sample'] == sample_t] for sample_t in samples}
for s in coords_raw:
    coords = coords_raw[s]
    uniq = np.unique(coords, axis=0)
    if uniq.shape[0] != coords.shape[0]:
        print(f"[{s}] duplicated coords:", coords.shape[0] - uniq.shape[0])


coords_raw['45'] = coords_raw['0'].copy() #
coords_raw['45'][:, 0] = coords_raw['45'][:, 0] + 1000 #
coords_raw['45'][:, 1] = coords_raw['45'][:, 1] + 100 #
print(coords_raw['45'].min(0), coords_raw['45'].max(0)) #
exp_dict = {sample_t: adata[adata.obs['sample'] == sample_t].layers['norm_1e4'] for sample_t in samples} #
exp_dict['45'] = exp_dict['0'].copy() #
embed_dict = CAST.CAST_MARK(coords_raw,exp_dict,output_path,epoch_t=400)
CAST.kmeans_plot_multiple(embed_dict,samples,coords_raw,'demo1',output_path,k=20,dot_size = 10,minibatch=False)
graph_list = ['45','0'] # [query_sample, reference_sample]
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
    iterations_bs=[0],  # 只跑 Affine，不执行 FFD
    attention_params_bs=[[None, 3, 1, 0]],
    mesh_weight=[None]
)

params_dist.alpha_basis = torch.Tensor([1/2000, 1/2000, 1/100, 3, 3]).reshape(5, 1).to(params_dist.device)

# 运行训练
coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list, params_dist)
