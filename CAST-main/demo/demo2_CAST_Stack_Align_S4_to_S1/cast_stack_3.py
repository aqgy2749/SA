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
# graph_list = ['30','0'] # [query_sample, reference_sample]
# query_sample = graph_list[0]

samples = ['0', '15', '30', '45', '60']   
reference_sample = '0'

for query_sample in samples:
    if query_sample == reference_sample:
        continue

    print(f'Aligning {query_sample} to {reference_sample} ...')

    graph_list = [query_sample, reference_sample]
    query_sample = graph_list[0]

    params_dist = CAST.reg_params(
        dataname=query_sample,
        gpu=0 if torch.cuda.is_available() else -1,
        diff_step=1,
        iterations=3000,
        dist_penalty1=0.1,
        bleeding=200,
        d_list=[3, 2, 1, 1/2, 1/3],
        attention_params=[None, 3, 1, 0],

        #### FFD parameters
        dist_penalty2=[0],
        alpha_basis_bs=[500],
        meshsize=[8],
        iterations_bs=[0],
        attention_params_bs=[[None, 3, 1, 0]],
        mesh_weight=[None]
    )

    params_dist.alpha_basis = torch.Tensor(
        [1/2000, 1/2000, 1/100, 3, 3]
    ).reshape(5, 1).to(params_dist.device)

    coords_final = CAST.CAST_STACK(
        coords_raw=coords_raw,
        embed_dict=embed_dict,
        output_path=output_path,
        graph_list=graph_list,
        params_dist=params_dist
    )

    print(f'Finished aligning {query_sample} to {reference_sample}\n')
