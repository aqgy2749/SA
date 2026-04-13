import os
import torch
import numpy as np
import anndata as ad
import scanpy as sc
import CAST
import warnings
warnings.filterwarnings("ignore")
work_dir = '/p1/data/jtian/SA/05_CAST' #### input the demo path

output_path = f'{work_dir}/output_mark'
os.makedirs(output_path, exist_ok=True)
adata = ad.read_h5ad(f'{output_path}/demo1.h5ad')
adata.layers['norm_1e4'] = adata.X
#adata.layers['norm_1e4'] = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X'].toarray() # we use normalized counts for each cell as input gene expression
######我们的数据不做归一化，我们数据也不是稀疏矩阵，不需要toarray转数组
samples = np.unique(adata.obs['sample']) # used samples in adata
coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs['sample'] == sample_t] for sample_t in samples}
exp_dict = {sample_t: adata[adata.obs['sample'] == sample_t].layers['norm_1e4'] for sample_t in samples}
embed_dict = torch.load("/p1/data/jtian/SA/05_CAST/output/demo_embed_dict.pt",map_location=torch.device('cpu'))
CAST.kmeans_plot_multiple(embed_dict,samples,coords_raw,'demo1',output_path,k=20,dot_size = 10,minibatch=False)