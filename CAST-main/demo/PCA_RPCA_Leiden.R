## =========================================================
## Seurat RPCA workflow for your lipid/spatial slices
## QC by slice -> merge all slices -> PCA -> RPCA integration
## -> neighbors on integrated embedding -> Leiden -> UMAP
## =========================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(ggplot2)
  library(patchwork)
  library(reticulate)
})

use_python("/p1/zulab_users/jtian/anaconda3/envs/r_vscode/bin/python", required = TRUE)
py_config()
py_module_available("leidenalg")
## -----------------------------
## 0. 基本参数
## -----------------------------
base_dir <- "/p2/zulab/jtian/data/SA/05_CAST/input"
out_dir  <- "/p2/zulab/jtian/data/SA/06_calculateConcentration/output_PCA_RPCA_Leiden/"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## 如果你的 csv 实际分隔符不是逗号，而是分号，就改成 ";"
csv_sep <- ";"

samples <- c("ctrl1", "ctrl2", "ctrl3", "dn1", "dn2", "dn3")
slices  <- c(0, 15, 30, 45, 60)

## 下游参数
nfeatures_use <- 3000
npcs_use      <- 50
dims_use      <- 1:30
resolution_use <- 1.9

# ## -----------------------------
# ## 1. 构建 sample sheet
# ## -----------------------------
# sample_sheet <- expand.grid(
#   sample = samples,
#   slice  = slices,
#   stringsAsFactors = FALSE
# )

# sample_sheet <- sample_sheet[order(match(sample_sheet$sample, samples),
#                                    sample_sheet$slice), ]

# sample_sheet$condition  <- ifelse(grepl("^ctrl", sample_sheet$sample), "ctrl", "dn")
# sample_sheet$replicate  <- sub("^[A-Za-z]+", "", sample_sheet$sample)
# sample_sheet$library_id <- paste0(sample_sheet$sample, "_slice", sample_sheet$slice)

# sample_sheet$intensity_file <- file.path(
#   base_dir, "LipidsIntensity", sample_sheet$sample,
#   paste0("lipid", sample_sheet$slice, ".csv")
# )

# sample_sheet$coord_file <- file.path(
#   base_dir, "LipidsSpotsIndex", sample_sheet$sample,
#   paste0("lipids", sample_sheet$slice, "RegionSpots.csv")
# )

# write.csv(sample_sheet,
#           file = file.path(out_dir, "sample_sheet.csv"),
#           row.names = FALSE)

# ## -----------------------------
# ## 2. 一些辅助函数
# ## -----------------------------

# ## 读取 csv；row_names = TRUE 时把第一列当作行名
# read_text_table <- function(file, row_names = FALSE, sep = ",") {
#   if (row_names) {
#     df <- read.table(
#       file = file,
#       header = TRUE,
#       sep = sep,
#       row.names = 1,
#       check.names = FALSE,
#       stringsAsFactors = FALSE
#     )
#   } else {
#     df <- read.table(
#       file = file,
#       header = TRUE,
#       sep = sep,
#       check.names = FALSE,
#       stringsAsFactors = FALSE
#     )
#   }
#   return(df)
# }

# ## 读取 intensity 矩阵：行=feature，列=cell
# read_intensity_matrix <- function(file, sep = ",") {
#   df <- read_text_table(file, row_names = TRUE, sep = sep)
#   mat <- as.matrix(df)
#   storage.mode(mat) <- "numeric"

#   if (any(is.na(mat))) {
#     stop(paste("强度矩阵中存在 NA，请先处理：", file))
#   }
#   if (any(mat < 0)) {
#     stop(paste("强度矩阵中存在负值。下面这套 CreateSeuratObject(counts=...) 流程默认要求非负值：", file))
#   }

#   mat <- Matrix(mat, sparse = TRUE)
#   return(mat)
# }

# ## 自动识别坐标列；优先找名字像 x/y 的列，否则默认前两列
# standardize_coords <- function(coord_df, cell_names_original) {
#   nm_low <- tolower(colnames(coord_df))

#   x_candidates <- which(nm_low %in% c(
#     "x", "coord_x", "xcoord", "pos_x", "posx", "imagecol", "col", "pxl_col_in_fullres"
#   ))
#   y_candidates <- which(nm_low %in% c(
#     "y", "coord_y", "ycoord", "pos_y", "posy", "imagerow", "row", "pxl_row_in_fullres"
#   ))

#   if (length(x_candidates) >= 1 && length(y_candidates) >= 1) {
#     xcol <- x_candidates[1]
#     ycol <- y_candidates[1]
#   } else {
#     if (ncol(coord_df) < 2) {
#       stop("坐标文件列数少于2，无法识别 x/y")
#     }
#     xcol <- 1
#     ycol <- 2
#   }

#   coord_out <- data.frame(
#     x = as.numeric(coord_df[[xcol]]),
#     y = as.numeric(coord_df[[ycol]]),
#     stringsAsFactors = FALSE
#   )

#   ## 尝试找 cell id 列
#   id_candidates <- which(nm_low %in% c(
#     "cell", "cell_id", "barcode", "spot", "spot_id", "id"
#   ))

#   if (length(id_candidates) >= 1) {
#     rownames(coord_out) <- as.character(coord_df[[id_candidates[1]]])
#   }

#   ## 如果坐标里有 cell id，并且能匹配 intensity 的列名，则按列名重排
#   if (!is.null(rownames(coord_out)) && all(cell_names_original %in% rownames(coord_out))) {
#     coord_out <- coord_out[cell_names_original, , drop = FALSE]
#   } else {
#     ## 否则要求行数和细胞数相同，按原顺序对齐
#     if (nrow(coord_out) != length(cell_names_original)) {
#       stop(paste0(
#         "坐标行数与 intensity 的细胞列数不一致，且未找到可匹配的 cell id。\n",
#         "坐标行数 = ", nrow(coord_out),
#         ", intensity细胞数 = ", length(cell_names_original)
#       ))
#     }
#     rownames(coord_out) <- cell_names_original
#   }

#   return(coord_out)
# }

# ## 构建“一个切片 = 一个 Seurat object”
# make_slice_object <- function(intensity_file, coord_file,
#                               sample, condition, replicate, slice_id,
#                               sep = ",") {

#   mat <- read_intensity_matrix(intensity_file, sep = sep)

#   cell_names_original <- colnames(mat)
#   if (is.null(cell_names_original)) {
#     cell_names_original <- paste0("cell", seq_len(ncol(mat)))
#     colnames(mat) <- cell_names_original
#   }

#   coord_df <- read_text_table(coord_file, row_names = FALSE, sep = sep)
#   coord_std <- standardize_coords(coord_df, cell_names_original = cell_names_original)

#   ## 给所有细胞加唯一前缀，避免后续合并重名
#   cell_names_new <- make.unique(
#     paste(sample, paste0("slice", slice_id), cell_names_original, sep = "_")
#   )
#   colnames(mat) <- cell_names_new
#   rownames(coord_std) <- cell_names_new

#   meta <- data.frame(
#     sample     = sample,
#     condition  = condition,
#     replicate  = replicate,
#     slice_id   = as.character(slice_id),
#     library_id = paste0(sample, "_slice", slice_id),
#     x          = coord_std$x,
#     y          = coord_std$y,
#     row.names  = cell_names_new,
#     stringsAsFactors = FALSE
#   )

#   obj <- CreateSeuratObject(
#     counts = mat,
#     assay = "LIPID",
#     project = "mouse_lipid",
#     min.cells = 0,
#     min.features = 0,
#     meta.data = meta
#   )

#   return(obj)
# }

# ## 计算稳健的 QC 边界：中位数 ± 3*MAD
# get_bounds_mad <- function(v, nmads = 3) {
#   v <- as.numeric(v)
#   med <- median(v, na.rm = TRUE)
#   md  <- mad(v, na.rm = TRUE)

#   if (is.na(md) || md == 0) {
#     q <- as.numeric(quantile(v, probs = c(0.01, 0.99), na.rm = TRUE))
#     return(c(max(0, q[1]), q[2]))
#   } else {
#     lower <- max(0, med - nmads * md)
#     upper <- med + nmads * md
#     return(c(lower, upper))
#   }
# }

# ## 按切片 QC
# qc_filter_one_slice <- function(obj, nmads = 3) {
#   assay_name <- DefaultAssay(obj)
#   meta <- obj[[]]

#   ncount_col   <- paste0("nCount_", assay_name)
#   nfeature_col <- paste0("nFeature_", assay_name)

#   nc <- meta[[ncount_col]]
#   nf <- meta[[nfeature_col]]

#   count_bounds   <- get_bounds_mad(nc, nmads = nmads)
#   feature_bounds <- get_bounds_mad(nf, nmads = nmads)

#   keep <- nc >= count_bounds[1] & nc <= count_bounds[2] &
#           nf >= feature_bounds[1] & nf <= feature_bounds[2]

#   ## 如果过严导致保留太少，自动回退到分位数过滤
#   if (sum(keep) < max(50, round(length(keep) * 0.5))) {
#     count_bounds   <- as.numeric(quantile(nc, probs = c(0.01, 0.99), na.rm = TRUE))
#     feature_bounds <- as.numeric(quantile(nf, probs = c(0.01, 0.99), na.rm = TRUE))

#     keep <- nc >= count_bounds[1] & nc <= count_bounds[2] &
#             nf >= feature_bounds[1] & nf <= feature_bounds[2]
#   }

#   if (sum(keep) == 0) {
#     stop(paste("这个切片 QC 后 0 个细胞，请检查阈值或原始数据：", unique(meta$library_id)))
#   }

#   before_df <- data.frame(
#     library_id      = unique(meta$library_id),
#     sample          = unique(meta$sample),
#     condition       = unique(meta$condition),
#     replicate       = unique(meta$replicate),
#     slice_id        = unique(meta$slice_id),
#     n_cells         = ncol(obj),
#     median_nCount   = median(nc, na.rm = TRUE),
#     median_nFeature = median(nf, na.rm = TRUE),
#     nCount_low      = count_bounds[1],
#     nCount_high     = count_bounds[2],
#     nFeature_low    = feature_bounds[1],
#     nFeature_high   = feature_bounds[2],
#     stringsAsFactors = FALSE
#   )

#   obj_filt <- subset(obj, cells = colnames(obj)[keep])

#   meta2 <- obj_filt[[]]
#   after_df <- data.frame(
#     library_id      = unique(meta2$library_id),
#     sample          = unique(meta2$sample),
#     condition       = unique(meta2$condition),
#     replicate       = unique(meta2$replicate),
#     slice_id        = unique(meta2$slice_id),
#     n_cells         = ncol(obj_filt),
#     median_nCount   = median(meta2[[ncount_col]], na.rm = TRUE),
#     median_nFeature = median(meta2[[nfeature_col]], na.rm = TRUE),
#     stringsAsFactors = FALSE
#   )

#   return(list(obj = obj_filt, before = before_df, after = after_df))
# }

# ## -----------------------------
# ## 3. 逐切片读取 + 按切片 QC
# ## -----------------------------
# obj.list <- list()
# qc.before.list <- list()
# qc.after.list  <- list()

# for (i in seq_len(nrow(sample_sheet))) {
#   rowi <- sample_sheet[i, ]

#   if (!file.exists(rowi$intensity_file)) {
#     stop(paste("找不到 intensity 文件：", rowi$intensity_file))
#   }
#   if (!file.exists(rowi$coord_file)) {
#     stop(paste("找不到坐标文件：", rowi$coord_file))
#   }

#   cat("读取并QC:", rowi$library_id, "\n")

#   obj0 <- make_slice_object(
#     intensity_file = rowi$intensity_file,
#     coord_file     = rowi$coord_file,
#     sample         = rowi$sample,
#     condition      = rowi$condition,
#     replicate      = rowi$replicate,
#     slice_id       = rowi$slice,
#     sep            = csv_sep
#   )

#   qc_res <- qc_filter_one_slice(obj0, nmads = 3)

#   obj.list[[rowi$library_id]]       <- qc_res$obj
#   qc.before.list[[rowi$library_id]] <- qc_res$before
#   qc.after.list[[rowi$library_id]]  <- qc_res$after
# }

# qc.before.df <- do.call(rbind, qc.before.list)
# qc.after.df  <- do.call(rbind, qc.after.list)

# write.csv(qc.before.df,
#           file = file.path(out_dir, "qc_before_by_slice.csv"),
#           row.names = FALSE)
# write.csv(qc.after.df,
#           file = file.path(out_dir, "qc_after_by_slice.csv"),
#           row.names = FALSE)

# saveRDS(obj.list, file = file.path(out_dir, "obj_list_after_qc.rds"))

# ## -----------------------------
# ## 4. 先合并全部切片（未整合对象）
# ##    这一步是你要求的 “合并全部切片 -> PCA”
# ## -----------------------------
# cat("Merging all slices...\n")

# obj.merged <- Reduce(function(x, y) merge(x = x, y = y), obj.list)

# ## 未整合数据做标准预处理
# DefaultAssay(obj.merged) <- "LIPID"

# obj.merged <- NormalizeData(
#   obj.merged,
#   normalization.method = "LogNormalize",
#   scale.factor = 10000,
#   verbose = FALSE
# )

# obj.merged <- FindVariableFeatures(
#   obj.merged,
#   selection.method = "vst",
#   nfeatures = nfeatures_use,
#   verbose = FALSE
# )

# obj.merged <- ScaleData(obj.merged, verbose = FALSE)

# obj.merged <- RunPCA(
#   obj.merged,
#   npcs = npcs_use,
#   reduction.name = "pca_unintegrated",
#   verbose = FALSE
# )

# saveRDS(obj.merged, file = file.path(out_dir, "merged_unintegrated_pca.rds"))

# pdf(file.path(out_dir, "elbow_unintegrated.pdf"), width = 7, height = 5)
# print(ElbowPlot(obj.merged, ndims = npcs_use, reduction = "pca_unintegrated"))
# dev.off()

# ## -----------------------------
# ## 5. RPCA 整合
# ##    注意：RPCA 不是在 merge 后直接做
# ##    而是要回到“每个切片一个对象”的 list
# ## -----------------------------
# cat("Running RPCA integration...\n")

# obj.list.rpca <- lapply(obj.list, function(x) {
#   DefaultAssay(x) <- "LIPID"

#   x <- NormalizeData(
#     x,
#     normalization.method = "LogNormalize",
#     scale.factor = 10000,
#     verbose = FALSE
#   )

#   x <- FindVariableFeatures(
#     x,
#     selection.method = "vst",
#     nfeatures = nfeatures_use,
#     verbose = FALSE
#   )

#   return(x)
# })

# ## 选用于 integration 的共同高变特征
# integration_features <- SelectIntegrationFeatures(
#   object.list = obj.list.rpca,
#   nfeatures = nfeatures_use
# )

# ## RPCA 前：每个切片单独 Scale + PCA
# obj.list.rpca <- lapply(obj.list.rpca, function(x) {
#   x <- ScaleData(x, features = integration_features, verbose = FALSE)
#   x <- RunPCA(
#     x,
#     features = integration_features,
#     npcs = npcs_use,
#     verbose = FALSE
#   )
#   return(x)
# })

# ## 找锚点：reduction = "rpca"
# anchors <- FindIntegrationAnchors(
#   object.list = obj.list.rpca,
#   anchor.features = integration_features,
#   reduction = "rpca",
#   dims = dims_use,
#   k.anchor = 5,
#   verbose = TRUE
# )

# ## 整合
# obj.integrated <- IntegrateData(
#   anchorset = anchors,
#   dims = dims_use,
#   verbose = TRUE
# )

# saveRDS(obj.integrated, file = file.path(out_dir, "integrated_raw.rds"))

obj.integrated <- readRDS(file.path(out_dir, "integrated_raw.rds"))

## -----------------------------
## 6. 用整合后的 assay / embedding 做 PCA、neighbors、Leiden、UMAP
## -----------------------------
DefaultAssay(obj.integrated) <- "integrated"

obj.integrated <- ScaleData(obj.integrated, verbose = FALSE)

obj.integrated <- RunPCA(
  obj.integrated,
  npcs = npcs_use,
  reduction.name = "integrated_pca",
  verbose = FALSE
)

pdf(file.path(out_dir, "elbow_integrated.pdf"), width = 7, height = 5)
print(ElbowPlot(obj.integrated, ndims = npcs_use, reduction = "integrated_pca"))
dev.off()

## 在整合后的 embedding 上建图
obj.integrated <- FindNeighbors(
  obj.integrated,
  reduction = "integrated_pca",
  dims = dims_use,
  verbose = FALSE
)

## Leiden 聚类
## graph.name 一般就是 integrated_snn，因为当前 DefaultAssay = "integrated"
obj.integrated <- FindClusters(
  obj.integrated,
  graph.name = "integrated_snn",
  cluster.name = "leiden",
  algorithm = 4,
  resolution = resolution_use,
  verbose = TRUE
)

Idents(obj.integrated) <- obj.integrated$leiden

## UMAP
obj.integrated <- RunUMAP(
  obj.integrated,
  reduction = "integrated_pca",
  dims = dims_use,
  reduction.name = "umap.rpca",
  return.model = TRUE,
  verbose = FALSE
)

## 保存最终对象
saveRDS(obj.integrated, file = file.path(out_dir, "seurat_rpca_final.rds"))
write.csv(obj.integrated[[]],
          file = file.path(out_dir, "final_metadata.csv"))

## -----------------------------
## 7. 简单结果图
## -----------------------------
p_umap_condition <- DimPlot(
  obj.integrated,
  reduction = "umap.rpca",
  group.by = "condition"
) + ggtitle("UMAP by condition")

p_umap_sample <- DimPlot(
  obj.integrated,
  reduction = "umap.rpca",
  group.by = "sample",
  label = FALSE
) + ggtitle("UMAP by sample")

p_umap_slice <- DimPlot(
  obj.integrated,
  reduction = "umap.rpca",
  group.by = "slice_id",
  label = FALSE
) + ggtitle("UMAP by slice")

p_umap_cluster <- DimPlot(
  obj.integrated,
  reduction = "umap.rpca",
  group.by = "leiden",
  label = TRUE
) + ggtitle("UMAP by Leiden cluster")

ggsave(
  filename = file.path(out_dir, "umap_condition_sample.png"),
  plot = p_umap_condition + p_umap_sample,
  width = 12, height = 5, dpi = 300
)

ggsave(
  filename = file.path(out_dir, "umap_slice_cluster.png"),
  plot = p_umap_slice + p_umap_cluster,
  width = 12, height = 5, dpi = 300
)

cat("Done. Final object saved to:\n",
    file.path(out_dir, "seurat_rpca_final.rds"), "\n")
## =========================================================
## 8. 在原始空间坐标上按簇上色
##    6个组，每组一张图，图内含 0/15/30/45/60 五个切片
## =========================================================

## 从最终对象中提取元数据
plot_df <- obj.integrated[[]]

## 确保有这些列
required_cols <- c("sample", "slice_id", "x", "y", "leiden")
missing_cols <- setdiff(required_cols, colnames(plot_df))
if (length(missing_cols) > 0) {
  stop(paste("缺少以下作图所需列：", paste(missing_cols, collapse = ", ")))
}

## 统一顺序
plot_df$sample <- factor(
  plot_df$sample,
  levels = c("ctrl1", "ctrl2", "ctrl3", "dn1", "dn2", "dn3")
)

plot_df$slice_id <- factor(
  as.character(plot_df$slice_id),
  levels = c("0", "15", "30", "45", "60")
)

plot_df$leiden <- factor(plot_df$leiden)

## 如果你的坐标原点在左上角（很多图像坐标是这样），
## 用 scale_y_reverse() 会更接近原始切片方向。
## 如果发现上下颠倒，就保留 scale_y_reverse()；
## 如果发现方向本来就是对的，就把这一行删掉。

for (smp in levels(plot_df$sample)) {
  df_sub <- subset(plot_df, sample == smp)

  p_spatial <- ggplot(df_sub, aes(x = x, y = y, color = leiden)) +
    geom_point(size = 0.35, alpha = 0.9) +
    facet_wrap(~ slice_id, nrow = 1) +
    coord_fixed() +
    scale_y_reverse() +
    theme_bw(base_size = 12) +
    labs(
      title = paste0("Spatial clusters - ", smp),
      x = "X",
      y = "Y",
      color = "Leiden"
    ) +
    theme(
      panel.grid = element_blank(),
      strip.background = element_rect(fill = "white"),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text = element_blank(),
      axis.ticks = element_blank()
    )

  ggsave(
    filename = file.path(out_dir, paste0("spatial_clusters_", smp, ".png")),
    plot = p_spatial,
    width = 20,
    height = 4.5,
    dpi = 300
  )

  pdf(
    file = file.path(out_dir, paste0("spatial_clusters_", smp, ".pdf")),
    width = 20,
    height = 4.5
  )
  print(p_spatial)
  dev.off()
}
p_all <- ggplot(plot_df, aes(x = x, y = y, color = leiden)) +
  geom_point(size = 0.25, alpha = 0.9) +
  facet_grid(sample ~ slice_id) +
  coord_fixed() +
  scale_y_reverse() +
  theme_bw(base_size = 10) +
  labs(
    title = "Spatial clusters of all samples and slices",
    x = "X",
    y = "Y",
    color = "Leiden"
  ) +
  theme(
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "white"),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )

ggsave(
  filename = file.path(out_dir, "spatial_clusters_all_samples.png"),
  plot = p_all,
  width = 18,
  height = 16,
  dpi = 300
)