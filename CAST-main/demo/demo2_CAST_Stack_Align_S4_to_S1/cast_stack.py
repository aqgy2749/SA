import CAST
import os, torch
import warnings
import anndata as ad
import numpy as np
warnings.filterwarnings("ignore")
work_dir = '/p1/data/jtian/SA/05_CAST' #### input the demo path
output_path = f'{work_dir}/output'
os.makedirs(output_path,exist_ok = True)
adata = ad.read_h5ad(f'{output_path}/demo1.h5ad')
samples = np.unique(adata.obs['sample'])
coords_raw = {s: np.array(adata.obs[['x','y']])[adata.obs['sample'] == s] for s in samples}
# coords_raw = torch.load(f'{output_path}/demo2_coords_raw.pt',map_location='cpu')
embed_dict = torch.load(f'{output_path}/demo_embed_dict.pt',map_location='cpu')
graph_list = ['15','0'] # [query_sample, reference_sample]
query_sample = graph_list[0]
params_dist = CAST.reg_params(
    dataname=query_sample,
    gpu=0 if torch.cuda.is_available() else -1,
    diff_step=1,  # 减小步长，帮助更平稳的收敛
    iterations=1, #3000,  # 增加迭代次数，以便更多的训练
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

# params_dist.alpha_basis = torch.Tensor([1/2000, 1/2000, 1/100, 3, 3]).reshape(5, 1).to(params_dist.device)

# 运行训练
coords_final = CAST.CAST_STACK(coords_raw, embed_dict, output_path, graph_list, params_dist)
