import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "0"


import CAST
import scanpy as sc
import os
import numpy as np
import warnings
import dgl
import torch
import CAST
import os
import numpy as np
import anndata as ad
import scanpy as sc
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
work_dir = '/p1/data/jtian/SA/05_CAST' #### input the demo path
output_path = f'{work_dir}/output_castProj1'
os.makedirs(output_path,exist_ok=True)
graph_list = {'rep1':['15','0']} # source_sample, target_sample
import scanpy as sc
import numpy as np
import torch

# # 读数据
# sdata = sc.read_h5ad("/p1/data/jtian/SA/05_CAST/output_demo3_CAST_project/demo1_castProj2.h5ad")

# # ========= 1) 表达层准备 =========
# # sdata.layers['raw'] = sdata.X.copy()
# sdata.layers['log1p_norm'] = sdata.X.copy()
# sdata.layers['log1p_norm_scaled'] = sdata.X.copy()

# # ========= 2) 读取对齐坐标 =========
# coords = torch.load("/p1/data/jtian/SA/05_CAST/output_cast9/15_align_to_0_coords_final.data", map_location='cpu')

# # 统一把 coords[key] 转成 numpy（兼容 torch.Tensor / numpy.ndarray）
# def to_numpy(a):
#     if isinstance(a, torch.Tensor):
#         return a.detach().cpu().numpy()
#     return np.asarray(a)

# c15 = to_numpy(coords['15'])
# c0  = to_numpy(coords['0'])

# mask15 = sdata.obs['sample'].astype(str) == '15'
# mask0  = sdata.obs['sample'].astype(str) == '0'

# # 确保列存在
# if 'x' not in sdata.obs.columns:
#     sdata.obs['x'] = np.nan
# if 'y' not in sdata.obs.columns:
#     sdata.obs['y'] = np.nan

# # ✅ 强烈建议：先做数量一致性检查（不一致就不要写入）
# assert c15.shape[0] == mask15.sum(), f"coords['15']行数={c15.shape[0]} 但 sample=15 的细胞数={mask15.sum()}"
# assert c0.shape[0]  == mask0.sum(),  f"coords['0']行数={c0.shape[0]} 但 sample=0 的细胞数={mask0.sum()}"

# # 写入对齐坐标
# sdata.obs.loc[mask15, 'x'] = c15[:, 0]
# sdata.obs.loc[mask15, 'y'] = c15[:, 1]

# sdata.obs.loc[mask0, 'x'] = c0[:, 0]
# sdata.obs.loc[mask0, 'y'] = c0[:, 1]
# #PCA+UMAP
# # PCA（不依赖 highly_variable）
# sc.tl.pca(sdata, svd_solver='full', n_comps=50)
# # 邻居图 + UMAP（用 PCA）
# sc.pp.neighbors(sdata, n_neighbors=50, n_pcs=30, use_rep='X_pca')
# sc.tl.umap(sdata, min_dist=0.01, spread=5)
# sc.settings.figdir = output_path   
# sc.pl.umap(
#     sdata,
#     color=['sample'],
#     size=5,
#     save="_umap_by_sample.png"     
# )
# sdata.write_h5ad(f'{output_path}/demo1_castProj1.h5ad')


# ========= 3) CAST preprocess =========
sdata = sc.read_h5ad(f'{output_path}/demo1_castProj1.h5ad')
# sdata = CAST.preprocess_fast(sdata, mode='default')
batch_key = 'sample'

sdata_refs = {}
list_ts = {}
color_dict = {
        'TEPN': '#256b00',
        'INH': '#ee750a',
        'CHO_PEP': '#f280cf',
        'DE_MEN': '#f24f4b',
        'AC': '#e8e879',
        'OLG': '#a8e1eb',
        'VAS': '#395ba8',
        'CHOR_EPEN': '#697491',
        'PVM': '#8803fc',
        'MLG': '#23ccb8',
        'OPC': '#667872',
        'Other': '#ebebeb'
    }
for rep in graph_list.keys():
    print(f'Start the {rep} samples:')
    source_sample, target_sample = graph_list[rep]
    output_path_t = f'{output_path}/{source_sample}_to_{target_sample}'
    os.makedirs(output_path_t,exist_ok=True)
    sdata_refs[rep],list_ts[rep] = CAST.CAST_PROJECT(
        sdata_inte = sdata[np.isin(sdata.obs[batch_key],[source_sample, target_sample])], # the integrated dataset
        source_sample = source_sample, # the source sample name
        target_sample = target_sample, # the target sample name
        coords_source = np.array(sdata[np.isin(sdata.obs[batch_key],source_sample),:].obs.loc[:,['x','y']]), # the coordinates of the source sample
        coords_target = np.array(sdata[np.isin(sdata.obs[batch_key],target_sample),:].obs.loc[:,['x','y']]), # the coordinates of the target sample
        scaled_layer = 'log1p_norm_scaled', # the scaled layer name in `adata.layers`, which is used to be integrated
        batch_key = batch_key, # the column name of the samples in `obs`
        # source_sample_ctype_col = 'cell_type', # the column name of the cell type in `obs`
        source_sample_ctype_col = None,
        output_path = output_path_t, # the output path
        integration_strategy = None, # 'Harmony' or None (use existing integrated features)
        pc_feature = 'X_pca', 
        umap_feature = 'X_umap',
        color_dict = color_dict ,# the color dict for the cell type
        use_highly_variable_t = False
    )