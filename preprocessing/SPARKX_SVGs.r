# install.packages('devtools')
# devtools::install_github('xzhoulab/SPARK')
library(Seurat)
library(ggplot2)
library(patchwork)
library(SPARK)

ST_tissue_list <- list('sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6')
ST_root_path <-  '../dataset/'

for(i in 1:length(ST_tissue_list))
{
  ST_tissue <- ST_tissue_list[i]
  print(ST_tissue)
  expr_path <- paste0(ST_root_path, ST_tissue, "/filtered_feature_bc_matrix", collapse = NULL)
  spatial_path <- paste0(ST_root_path, ST_tissue, "/spatial", collapse = NULL)
  expr.mydata <- Seurat::Read10X(data.dir = expr_path)
  mydata <- Seurat::CreateSeuratObject(counts = expr.mydata, project = 'CRC', assay = 'Spatial')
  img <- Seurat::Read10X_Image(image.dir = spatial_path)
  Seurat::DefaultAssay(object = img) <- 'Spatial'
  img <- img[colnames(x = mydata)]
  mydata[['image']] <- img
  
  coordinates <- mydata@images$image@coordinates
  loc <- coordinates[, c("row","col")]
  colnames(loc) <- c("x","y")
  loc <- as.matrix(loc)
  counts <- as.matrix(mydata@assays$Spatial@counts)

  mt_idx <- grep("^MT-",rownames(counts))
  print(paste0("Number of mitochondrial gene:", length(mt_idx)))
  if(length(mt_idx)!=0){
      counts <- counts[-mt_idx,]
  }

  rb_idx <- grep("^RP[SL]",rownames(counts))
  print(paste0("Number of ribosomal gene:", length(rb_idx)))
  if(length(rb_idx)!=0){
      counts <- counts[-rb_idx,]
  }

  sparkX <- sparkx(counts,loc,numCores=16,option="mixture")
  
  sparkX_pvalue <- sparkX$res_mtest
  sparkX_pvalue_rank <- sparkX_pvalue[order(sparkX_pvalue$combinedPval),]
  
  SVG_top2000 <- row.names(sparkX_pvalue_rank[1:2000,])
  write.table(SVG_top2000, paste0('../preprocessed_data/SVG_top2000/',ST_tissue,'_SPARKX.txt'), row.names=FALSE, col.names = FALSE, quote=FALSE)
  
}








