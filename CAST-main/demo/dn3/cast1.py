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

grp='dn3'
warnings.filterwarnings("ignore")
work_dir = f'/p2/zulab/jtian/data/SA/05_CAST/{grp}'
output_path = f'{work_dir}/output_cast1'      ############
os.makedirs(output_path, exist_ok=True)

###############################preprocess################################################
#1.构建Anndata
# 读入intensity文件
intensity0 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsIntensity/{grp}/lipid0.csv", sep=";", header = 0, index_col = 0)
intensity15 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsIntensity/{grp}/lipid15.csv", sep=";", header = 0, index_col = 0)
intensity30 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsIntensity/{grp}/lipid30.csv", sep=";", header = 0, index_col = 0)
intensity45 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsIntensity/{grp}/lipid45.csv", sep=";", header = 0, index_col = 0)
intensity60 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsIntensity/{grp}/lipid60.csv", sep=";", header = 0, index_col = 0)
intensity0.index.name ='mz'
intensity15.index.name ='mz'
intensity30.index.name ='mz'
intensity45.index.name ='mz'
intensity60.index.name ='mz'
#打印形状 (Shape)
print("--- 数据形状 (Shape) ---")
print(intensity0.shape)
print(intensity15.shape)
print(intensity30.shape)
print(intensity45.shape)
print(intensity60.shape)

#打印前几行数据 (Head)
print("\n--- 数据预览 (Head) ---")
print("样本 0:")
print(intensity0.head())
print("\n样本 15:")
print(intensity15.head())
print("\n样本 30:")
print(intensity30.head())
print("\n样本 45:")
print(intensity45.head())
print("\n样本 60:")
print(intensity60.head())

coords0 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsSpotsIndex/{grp}/lipids0RegionSpots.csv", sep=";", header=0, index_col=0)
coords15 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsSpotsIndex/{grp}/lipids15RegionSpots.csv", sep=";", header=0, index_col=0)
coords30 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsSpotsIndex/{grp}/lipids30RegionSpots.csv", sep=";", header=0, index_col=0)
coords45 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsSpotsIndex/{grp}/lipids45RegionSpots.csv", sep=";", header=0, index_col=0)
coords60 = pd.read_csv(f"/p2/zulab/jtian/data/SA/05_CAST/input/LipidsSpotsIndex/{grp}/lipids60RegionSpots.csv", sep=";", header=0, index_col=0)

coords0.index.name = 'SpotIndex'
coords15.index.name = 'SpotIndex'
coords30.index.name = 'SpotIndex'
coords45.index.name = 'SpotIndex'
coords60.index.name = 'SpotIndex'

# 打印形状 (Shape)
print("--- 数据形状 (Shape) ---")
print(coords0.shape)
print(coords15.shape)
print(coords30.shape)
print(coords45.shape)
print(coords60.shape)

# 打印前几行数据 (Head)
print("\n--- 数据预览 (Head) ---")
print("样本 0:")
print(coords0.head())
print(coords0.tail())
print("\n样本 15:")
print(coords15.head())
print(coords15.tail())
print("\n样本 30:")
print(coords30.head())
print(coords30.tail())
print("\n样本 45:")
print(coords45.head())
print(coords45.tail())
print("\n样本 60:")
print(coords60.head())
print(coords60.tail())

print("\n--- 所有文件读入完成 ---")

SAMPLE_IDS = [0, 15, 30, 45, 60]

# --- 1. 初始化容器 ---
all_X_t = []  # 存储转置后的特征矩阵 (Spot x Metabolite)
all_obs = []  # 存储所有的观测值 (坐标和样本ID)
all_var_names = None # 存储代谢物 mz 值/索引

print("开始整合数据...")

for sample_id in SAMPLE_IDS:
    intensity_df = globals()[f'intensity{sample_id}']
    coords_df = globals()[f'coords{sample_id}']

    print(f"\n整合样本 {sample_id}...")

    # --- 2. 处理特征矩阵 (创建 .X) ---
    # AnnData 要求：行是 Spot，列是 Metabolite。
    # 原始 intensity_df 形状是 (Metabolite, Spot)。需要转置 (T) 并提取值 (.values)。
    X_t = intensity_df.T.values
    all_X_t.append(X_t)
    obs_t = coords_df.loc[:, ['x', 'y']].copy()
    # 坐标向下取整
    obs_t[['x', 'y']] = np.floor(obs_t[['x', 'y']].to_numpy()).astype(int)       #########################
    obs_t['sample'] = str(sample_id)
    all_obs.append(obs_t)

    # --- 4. 处理变量名 (创建 .var) ---
    # AnnData 要求：.var 的索引是特征 ID (mz 值)
    if all_var_names is None:
        # 使用 mz 值作为所有样本共享的变量名 (特征 ID)
        all_var_names = intensity_df.index.astype(str)
        # 验证所有样本的 mz 值是否一致
        for next_sample_id in SAMPLE_IDS:
             next_intensity_df = globals()[f'intensity{next_sample_id}']
             if not all_var_names.equals(next_intensity_df.index.astype(str)):
                 print("⚠️ 警告：不同样本的 $m/z$ 值（代谢物）列表不完全一致。这将使用第一个样本的 $m/z$ 列表。")
                 break

# --- 5. 整合所有样本的数据 ---

if not all_X_t:
    print("❌ 错误：没有成功处理任何样本，无法创建 AnnData 对象。")
else:
    # 合并所有样本的特征矩阵（Spot x Metabolite）
    X_combined = np.vstack(all_X_t)
    
    # 合并所有样本的观测值（坐标和样本 ID）
    obs_combined = pd.concat(all_obs, axis=0)
    
    # 创建变量 (特征) DataFrame，索引是 $m/z$ 值
    var_df = pd.DataFrame(index=all_var_names)
    
    # --- 6. 构建最终 AnnData 对象 ---
    print("\n构建 AnnData 对象...")
    adata = ad.AnnData(
        X=X_combined,
        obs=obs_combined,
        var=var_df,
        dtype=X_combined.dtype
    )
    
    # --- 7. preprocess ---
    print("preprocess前")
    print(adata.X[:5])
    # #cheng10quzheng
    # adata.X = (adata.X * 10).round().astype(int)
    # print("×10取整")
    # print(adata.X[:5])
    # print(np.min(adata.X), np.max(adata.X))
    # #CPM
    # X = adata.X
    # n=5
    # scale = 10**n 
    # cell_sum = np.array(X.sum(axis=1)).flatten()
    # print(cell_sum.min(), cell_sum.max())
    # adata.X = (X / cell_sum[:, None]) * scale
    # print('CPM后')
    # print(adata.X[:5])
    # print(np.min(adata.X), np.max(adata.X))
    #log+scale
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False)
    print('log变换+scale')
    print(adata.X[:5])
    print(np.min(adata.X), np.max(adata.X))
    OUTPUT_H5AD = f'{output_path}/demo1_cast1.h5ad'                    ########################
    adata.write_h5ad(OUTPUT_H5AD)
    print(f"\n✅ 成功保存 AnnData 文件到 {OUTPUT_H5AD}")

###########################CAST.MARK##########################################
adata = ad.read_h5ad(f'{output_path}/demo1_cast1.h5ad')                #########################
adata.layers['norm_1e4'] = adata.X 
samples = np.unique(adata.obs['sample'])
coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs['sample'] == sample_t] for sample_t in samples}
# for s in samples:
#     anchor = coords_raw[s][0].copy()   
#     coords_raw[s] = coords_raw[s] - anchor
exp_dict = {sample_t: adata[adata.obs['sample'] == sample_t].layers['norm_1e4'] for sample_t in samples}
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
embed_dict = CAST_MARK(coords_raw, exp_dict, output_path, args=args)
GLOBAL_K = 20
_, global_label_dict = kmeans_plot_multiple(
    embed_dict,
    samples,
    coords_raw,
    'demo1',
    output_path,
    k=GLOBAL_K,
    dot_size=10,
    minibatch=False,
    return_label_dict=True,
    save_label_dict=True
)
# embed_dict = torch.load(f'{output_path}/demo_embed_dict.pt',map_location='cpu')#来自cast2
###########################CAST.STACK##########################################
graph_list = ['15','0']
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
                            iterations_bs = [0],
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

graph_list = ['30','15']
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
                            iterations_bs = [0],
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

graph_list = ['45','30']
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
                            iterations_bs = [0],
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

graph_list = ['60','45']
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
                            iterations_bs = [0],
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
