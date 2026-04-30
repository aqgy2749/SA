library(Seurat)
library(dplyr)
library(patchwork)
library(future)

# ===== 并行设置 =====
plan(multicore, workers = 4)
options(future.globals.maxSize = 100 * 1024^3)

# ===== 路径设置 =====
input_dir <- "/p2/zulab/jtian/data/SA/05_CAST/input/LipidsIntensity"
output_dir <- "/p2/zulab/jtian/data/SA/R-Integration/Integration1_fast"
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
  SelectIntegrationFeatures(object.list = seurat_list, nfeatures = 1000)
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
    dims = 1:10,
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
    dims = 1:10,
    normalization.method = "SCT"
  )
)

DefaultAssay(object = integrated) <- "integrated"
integrated <- load_or_run(
  "06_integrated_analysis",
  {
    integrated <- ScaleData(object = integrated, verbose = FALSE)
    integrated <- RunPCA(object = integrated, npcs = 15, verbose = FALSE)
    integrated <- FindNeighbors(integrated, dims = 1:10)
    integrated <- FindClusters(integrated, resolution = 0.4)
    integrated <- RunUMAP(
      object = integrated,
      reduction = "pca",
      dims = 1:10,
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
