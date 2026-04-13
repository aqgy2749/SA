import os
import torch
import numpy as np
import anndata as ad
from sklearn.cluster import KMeans
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
work_dir = '/p1/data/jtian/SA/05_CAST/'
output_path = f'{work_dir}/output_cast9'      ############
os.makedirs(output_path, exist_ok=True)

def load_cell_label_dict_from_output(output_path, k=20):
    # 1) 读回 MARK 保存的 embedding
    embed_dict = torch.load(os.path.join(output_path, "demo_embed_dict.pt"), map_location="cpu")  # :contentReference[oaicite:5]{index=5}

    # 2) 读回你保存的 h5ad，用来拿 samples 顺序 + 每个 sample 的spot数量
    adata = ad.read_h5ad(os.path.join(output_path, "demo1_cast9.h5ad"))  # :contentReference[oaicite:6]{index=6}
    samples = np.unique(adata.obs["sample"])

    coords_raw = {
        s: np.array(adata.obs[["x", "y"]])[adata.obs["sample"] == s]
        for s in samples
    }

    # 3) 按 kmeans_plot_multiple 的逻辑把 embedding row_stack 起来再做 KMeans
    #    （你当时 minibatch=False，所以是 KMeans，不是 MiniBatchKMeans）:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}
    embed_stack = embed_dict[samples[0]].cpu().detach().numpy()
    for s in samples[1:]:
        embed_stack = np.row_stack((embed_stack, embed_dict[s].cpu().detach().numpy()))

    labels_all = KMeans(n_clusters=k, random_state=0).fit(embed_stack).labels_

    # 4) 按每个 sample 的spot数切开
    cell_label_dict = {}
    offset = 0
    for s in samples:
        n = coords_raw[s].shape[0]
        cell_label_dict[s] = labels_all[offset: offset + n]
        offset += n

    return cell_label_dict

# 使用
cell_label_dict = load_cell_label_dict_from_output(output_path, k=20)

# （可选）这次把它真的存到 output 目录，之后就能“直接读”了
save_path = "/p1/data/jtian/SA/05_CAST/output_castProj1"
torch.save(cell_label_dict, os.path.join(save_path, "cell_label_dict_k20.pt"))
