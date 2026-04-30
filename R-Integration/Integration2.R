library(Seurat)
library(dplyr)
library(patchwork)
library(future)

# ===== 并行设置 =====
plan(multicore, workers = 4)
options(future.globals.maxSize = 100 * 1024^3)
set.seed(20260429)

# ===== 降采样设置 =====
# CSV 中行为 m/z，列为 Spot/cell；这里降采样的是列，不会减少 m/z 特征。
max_cells_per_sample <- 5000
min_total_intensity_quantile <- 0.05
n_intensity_strata <- 10

# ===== 路径设置 =====
input_dir <- "/p2/zulab/jtian/data/SA/05_CAST/input/LipidsIntensity"
output_dir <- "/p2/zulab/jtian/data/SA/R-Integration/Integration2_downsample_5000"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
checkpoint_dir <- file.path(output_dir, "checkpoints")
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

checkpoint_path <- function(name) {
  file.path(checkpoint_dir, paste0(name, ".rds"))
}

save_checkpoint <- function(object, name) {
  saveRDS(object, checkpoint_path(name))
  object
}

load_or_run <- function(name, expr) {
  path <- checkpoint_path(name)
  if (file.exists(path)) {
    message("Loading checkpoint: ", path)
    return(readRDS(path))
  }

  message("Running step: ", name)
  object <- force(expr)
  save_checkpoint(object, name)
}

downsample_spots <- function(data) {
  n_cells_before <- ncol(data)
  spot_total <- colSums(data, na.rm = TRUE)

  if (min_total_intensity_quantile > 0) {
    min_total <- as.numeric(
      quantile(spot_total, probs = min_total_intensity_quantile, na.rm = TRUE)
    )
    keep_by_intensity <- spot_total > min_total
    data <- data[, keep_by_intensity, drop = FALSE]
    spot_total <- spot_total[keep_by_intensity]
  }

  n_cells_after_filter <- ncol(data)
  if (!is.finite(max_cells_per_sample) || n_cells_after_filter <= max_cells_per_sample) {
    message(
      "cells before = ", n_cells_before,
      ", after intensity filter = ", n_cells_after_filter,
      ", after downsample = ", n_cells_after_filter
    )
    return(data)
  }

  breaks <- unique(quantile(
    spot_total,
    probs = seq(0, 1, length.out = n_intensity_strata + 1),
    na.rm = TRUE
  ))

  if (length(breaks) < 2) {
    keep_cells <- sort(sample(seq_len(n_cells_after_filter), max_cells_per_sample))
    data <- data[, keep_cells, drop = FALSE]
    message(
      "cells before = ", n_cells_before,
      ", after intensity filter = ", n_cells_after_filter,
      ", after random downsample = ", ncol(data)
    )
    return(data)
  }

  strata <- cut(
    spot_total,
    breaks = breaks,
    include.lowest = TRUE,
    labels = FALSE
  )

  cells_per_stratum <- table(strata)
  target_per_stratum <- pmax(
    1,
    round(max_cells_per_sample * cells_per_stratum / sum(cells_per_stratum))
  )

  keep_cells <- unlist(
    lapply(seq_along(cells_per_stratum), function(i) {
      stratum <- as.integer(names(cells_per_stratum)[i])
      idx <- which(strata == stratum)
      n_keep <- min(length(idx), as.integer(target_per_stratum[i]))
      sample(idx, n_keep)
    }),
    use.names = FALSE
  )

  if (length(keep_cells) > max_cells_per_sample) {
    keep_cells <- sample(keep_cells, max_cells_per_sample)
  }

  keep_cells <- sort(keep_cells)
  data <- data[, keep_cells, drop = FALSE]
  message(
    "cells before = ", n_cells_before,
    ", after intensity filter = ", n_cells_after_filter,
    ", after stratified downsample = ", ncol(data)
  )
  data
}

# ===== 1. 自动建立样本信息表 =====
# 输入目录结构：
# LipidsIntensity/
#   ctrl1/lipid0.csv, lipid15.csv, ...
#   ctrl2/lipid0.csv, lipid15.csv, ...
#   ...
# 分组 condition 直接使用文件夹名：ctrl1, ctrl2, ctrl3, dn1, dn2, dn3
sample_files <- list.files(
  path = input_dir,
  pattern = "^lipid[0-9]+\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)

sample_info <- data.frame(
  file = sample_files,
  condition = basename(dirname(sample_files)),
  file_name = basename(sample_files),
  stringsAsFactors = FALSE
)

sample_info$time <- as.integer(sub("^lipid([0-9]+)\\.csv$", "\\1", sample_info$file_name))
sample_info$sample <- paste0(sample_info$condition, "_", sample_info$time)
sample_info <- sample_info[order(sample_info$condition, sample_info$time), ]
rownames(sample_info) <- NULL

print(sample_info)

# ===== 2. 定义单个样本的分析函数 =====
process_sample <- function(file, sample, condition, time) {
  data <- read.csv(
    file = file,
    row.names = 1,
    header = TRUE,
    sep = ";",
    check.names = FALSE
  )

  data <- data * 10
  data <- round(data, digits = 0)
  data <- as.data.frame(data)
  rownames(data) <- make.unique(rownames(data))
  colnames(data) <- make.unique(gsub("\\s+", "_", colnames(data)))
  data <- downsample_spots(data)

  obj <- CreateSeuratObject(
    counts = data,
    project = "SA",
    min.features = 1
  )

  obj$Sample <- sample
  obj$Condition <- condition
  obj$Time <- time

  obj <- RenameCells(obj, add.cell.id = sample)

  obj <- SCTransform(obj, verbose = FALSE)
  obj <- RunPCA(obj, assay = "SCT", verbose = FALSE)

  obj <- FindNeighbors(obj, dims = 1:15)
  obj <- FindClusters(obj, resolution = 0.5)
  obj <- RunUMAP(
    obj,
    dims = 1:20,
    n.neighbors = 30L,
    min.dist = 0.1,
    check_duplicates = FALSE
  )

  return(obj)
}

# ===== 3. 批量处理所有样本 =====
seurat_list <- load_or_run(
  "01_seurat_list",
  lapply(
    seq_len(nrow(sample_info)),
    function(i) {
      process_sample(
        file = sample_info$file[i],
        sample = sample_info$sample[i],
        condition = sample_info$condition[i],
        time = sample_info$time[i]
      )
    }
  )
)

names(seurat_list) <- sample_info$sample

# ===== 4. SCT integration =====
features <- load_or_run(
  "02_integration_features",
  SelectIntegrationFeatures(object.list = seurat_list)
)

seurat_list <- load_or_run(
  "03_prepped_seurat_list",
  PrepSCTIntegration(
    object.list = seurat_list,
    anchor.features = features
  )
)

anchors <- load_or_run(
  "04_integration_anchors",
  FindIntegrationAnchors(
    object.list = seurat_list,
    dims = 1:20,
    anchor.features = features,
    normalization.method = "SCT",
    reduction = "rpca",
    k.anchor = 5
  )
)

integrated <- load_or_run(
  "05_integrated_data",
  IntegrateData(
    anchorset = anchors,
    dims = 1:20,
    normalization.method = "SCT"
  )
)

DefaultAssay(object = integrated) <- "integrated"
integrated <- load_or_run(
  "06_integrated_analysis",
  {
    integrated <- ScaleData(object = integrated, verbose = FALSE)
    integrated <- RunPCA(object = integrated, npcs = 20, verbose = FALSE)
    integrated <- FindNeighbors(integrated, dims = 1:13)
    integrated <- FindClusters(integrated, resolution = 0.4)
    integrated <- RunUMAP(
      object = integrated,
      reduction = "pca",
      dims = 1:13,
      n.neighbors = 20L,
      min.dist = 0.01,
      check_duplicates = FALSE
    )
    integrated
  }
)
head(x = Idents(object = integrated), 5)

# ===== 5. 输出结果 =====
saveRDS(
  integrated,
  file = file.path(output_dir, "integrated_lipids_20250625.rds")
)

umap_plot <- DimPlot(
  object = integrated,
  reduction = "umap",
  pt.size = 0.5,
  label.size = 5,
  label = TRUE,
  repel = TRUE
)

print(umap_plot)

data_to_write_out <- as.data.frame(as.matrix(integrated@active.ident))
write.csv(
  x = data_to_write_out,
  row.names = TRUE,
  file = file.path(output_dir, "cluster.csv")
)
