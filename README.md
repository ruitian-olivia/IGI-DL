## ST_IGI package
A python package called "ST_IGI", which is an integrated graph and image deep learning (IGI-DL) model for gene spatial expression prediction based on HE slides of cancer.

### Install ST_IGI package
```bash
install ST_IGI-0.1.0.tar.gz
```
### System requirement
Major dependencies are:
- python 3.9
- torch 1.11.0+cu102
- torchvision 0.12.0+cu102
- torch-cluster 1.6.0
- torch-scatter 2.0.9
- torch-sparse 0.6.15
- torch-geometric 2.1.0.post1
- opencv-python
- matplotlib
- pandas
- palettable

### Model weights
The model weights trained on three cancer types, cutaneous squamous cell carcinoma (cSCC), breast cancer,  and colorectal cancer (CRC), are saved in **./model_weights**.

### Demo data
Demo data can be downloaded on the [Zenodo](https://zenodo.org/record/7937635#.ZGMOF-xBy3J), including a cSCC sample sequenced by 10X Visium and a breast cancer sample sequenced by ST. Downloaded demo data can be saved in **./download_data**.

### Demo example
The usage example of this Python package is shown in the **ST_IGI_demo.ipynb**. When calling some functions of ST_IGI, it will automatically download the model weights and demo data. If it fails to download due to network reasons, you can download the relevant files yourself by referring to the contents of the two sections above, **Model weights** and **Demo data**.

## IGI-DL Training

Source code for Integrated Graph and Image Deep Learning (IGI-DL).

### Data structure

10X Visium spatial transcriptomics data with high-resolution HE-stained histological images are saved in **./dataset**.

```bash
dataset
└───sample1
│  └───HE_image
│    │  histology.jpg
│
│  └───filtered_feature_bc_matrix
│    │  barcodes.tsv.gz
│    │  features.tsv.gz
│    │  matrix.mtx.gz
│ 
│  └───spatial
│    │  aligned_fiducials.jpg
│    │  detected_tissue_image.jpg
│    │  scalefactors_json.json
│    │  tissue_hires_image.png
│    │  tissue_lowres_image.png
│    │  tissue_positions_list.csv
│ 
└───sample2
│  ...
```

**HE/** containes a high-resolution brightfield image stained with hematoxylin and eosin (H&E), which is before Space Ranger downsampling.

**filtered_feature_bc_matrix/** and  **spatial/** are the outputs of Space Ranger.
filtered_feature_bc_matrix: Feature barcode matrices that contains only tissue-associated barcodes.
spatial: Folder containing Visium-specific outs (QC images to check image processing pipeline, downsampled input images, and files that describe spot barcode locations in the images)

Preprocessed HE patches and constructed Nuclei-Graphs are saved in **./preprocessed_data**, where **HE_patches** contains HE patches extracted according to the coordinates of the spot, **HE_nmzd** containes patches after color normalization, **hover_seg** containes segmented nuclei in patches, **nuclei_seg_features** containes extracted nuclei features, **nuclei_standar_features** containes standardized nuclei features, **SVGs_label** containes transformed expression level for target genes, and **graph_image_pt** containes constructed Nuclei-Graphs.

```bash
preprocessed_data
└───HE_patches
│  └───sample1
│    │  ...
│  └───sample2
│    │  ...
│  └───sample3
│    │  ...
│
└───HE_nmzd
│  └───sample1
│    │  ...
│
└───hover_seg
│  └───sample1
│    │  ...
│
└───nuclei_seg_features
│  └───sample1
│    │  ...
│
└───nuclei_standar_features
│  └───sample1
│    │  ...
│
└───SVGs_label
│  └───sample1
│    │  ...
│
└───graph_image_pt
│  └───sample1
│    │  ...
```

### Data preprocessing
Code in **./preprocessing**

##### 1. Image preprocessing
###### 1.1 HE patches extraction

```bash
cd preprocessing
python patches_extract.py
```

###### 1.2 HE patches color normalization

```bash
cd preprocessing
python patches_normalization.py
```
Normalized HE patches are saved in **./preprocessed_data/HE_nmzd**.

###### 1.3 Nuclei segmentation

Using Hover-Nety<sup>[1]</sup> pretrained on PanNuke Datasety<sup>[2]</sup> to segment nucleus in the HE patches, where the model weight file is saved in **./Hover-Net/hovernet-inference-weights** as pannuke.npz.

```bash
mkdir Hover-Net
cd Hover-Net
git clone https://github.com/simongraham/hovernet_inference.git

python hovernet_inference/run.py \
--mode='tile' \
--model='./hovernet-inference-weights/pannuke.npz' \
--input_dir='../../preprocessed_data/HE_nmzd/sample1' \
--output_dir='../../preprocessed_data/hover_seg/sample1'

python hovernet_inference/run.py \
--mode='tile' \
--model='./hovernet-inference-weights/pannuke.npz' \
--input_dir='../../preprocessed_data/HE_nmzd/sample2' \
--output_dir='../../preprocessed_data/hover_seg/sample2'
```

###### 1.4 Nuclei features extraction

```bash
cd preprocessing
python nuclei_features_extract.py
```

###### 1.5 Nuclei features standardization

```bash
cd preprocessing
python nuclei_features_standardization.py
```

Standardized nuclei features for each patch are saves in **./preprocessed_data/nuclei_standar_features**.

##### 2. Gene expression preprocessing
###### 2.1 Find SVGs using SPARKX<sup>[3]</sup>

```bash
cd preprocessing
Rscript SPARKX_SVGs.r
```
SVGs for each tissue sample are saved in **./preprocessed_data/SVG_top2000**.

###### 2.2 Target genes selection

```bash
cd preprocessing
python target_genes_selection.py
```
Target genes list is saved in **SVGs_SPARKX.txt** .

###### 2.3 Count data transformation

```bash
cd preprocessing
python gene_count_tranform.py
```
Transformed count data for each tissue sample are saved in **./preprocessed_data/SVGs_label**.


##### 3. Nuclei-Graphs construction

```bash
cd preprocessing
python graph_construct.py
```

Constructed Nuclei-Graphs for patches in each tissue sample are saved in **./preprocessed_data/graph_SVGs**.

### Model training
Code in **./model_training**

##### Image-based models

- ResNet18
```bash
cd model_training/image_based
python resnet18_MLP_Y_main.py img_resnet18 2e-4 1e-4 300 30 --mlp_hidden 512 256 256
```

- ViT
```bash
cd model_training/image_based
python ViT_MLP_Y_main.py img_ViT 2e-4 1e-4 300 30 8 64 256 --mlp_hidden 512 256 256
```

##### Graph-based models

- GIN
```bash
cd model_training/graph_based
python gin4layer_multi_Y_main.py graph_GIN 2e-4 1e-4 256 300 30
```

- GCN
```bash
cd model_training/graph_based
python gcn4layer_multi_Y_main.py graph_GCN 2e-4 1e-4 256 300 30
```

- GAT
```bash
cd model_training/graph_based
python gat4layer_multi_Y_main.py graph_GAT 2e-4 1e-4 256 300 30
```

##### Integrated models

- GIN+ResNet18
```bash
cd model_training/integrated_model
python GIN_Res18_main.py integrated_GIN_resnet 2e-4 1e-4 256 300 30 --mlp_hidden 512 256 256
```

- GCN+ResNet18
```bash
cd model_training/integrated_model
python GCN_Res18_main.py integrated_GCN_resnet 2e-4 1e-4 256 300 30 --mlp_hidden 512 256 256
```

- GAT+ResNet18
```bash
cd model_training/integrated_model
python GAT_Res18_main.py integrated_GCN_resnet 2e-4 1e-4 256 300 30 --mlp_hidden 512 256 256
```

- GIN+ViT
```bash
cd model_training/integrated_model
python GIN_ViT_MLP_main.py integrated_GIN_ViT 2e-4 1e-4 256 300 30 8 64 256 256 --mlp_hidden 512 256 256
```
### IGI-DL prediction
Code in **./model_prediction**

##### Saving model weights
Model weights are saved as **./IGI-DL-weights.pth**
```bash
cd model_prediction
python IGI_training_main.py 2e-4 1e-4 256 300 30 --mlp_hidden 512 256 256
```

##### Predicting new samples
```bash
cd model_prediction
python IGI_test_main.py
```


### Reference

[1] Graham S, Vu Q D, Raza S E A, et al. Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images[J]. Medical Image Analysis, 2019, 58: 101563.

[2] Gamper J, Alemi Koohbanani N, Benet K, et al. Pannuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification[C]//European congress on digital pathology. Springer, Cham, 2019: 11-19.

[3] Zhu J, Sun S, Zhou X. SPARK-X: non-parametric modeling enables scalable and robust detection of spatial expression patterns for large spatial transcriptomic studies[J]. Genome Biology, 2021, 22(1): 1-25.