# Spatial-temporal model for historical map segmentation 

PyTorch implementation of "Cross-attention Spatio-temporal Context Transformer for Semantic Segmentation of Historical Maps".

## Abstract
Historical maps provide useful spatio-temporal information on the Earth’s surface before modern earth observation techniques came into being. To extract information from maps, neural networks, which gain wide popularity in recent years, have replaced hand-crafted map processing methods and tedious manual labor. However, aleatoric uncertainty, known as data-dependent uncertainty, inherent in the drawing/scanning/fading defects of the original map sheets and inadequate contexts when cropping maps into small tiles considering the memory limits of the training process, challenges the model to make correct predictions. As aleatoric uncertainty cannot be reduced even with more training data collected,
we argue that complementary spatio-temporal contexts can be helpful. To achieve this, we propose a U-Net-based network that fuses spatio-temporal features with cross-attention transformers(U-SpaTem), aggregating information at a larger spatial range as well as through a temporal sequence of images. Our model achieves a better performance than other state-or-art models that use either temporal or spatial contexts. Compared with pure vision transformers, our model is more lightweight and effective. To the best of our knowledge, leveraging both spatial and temporal contexts have been rarely explored before in the segmentation task. Even though our application is on segmenting historical maps, we believe that the method can be transferred into other fields with similar problems like temporal sequences of satellite images.

<embed src="img/overallarchitecturefinal.pdf" width="500" height="375" type="application/pdf">

If you find this code useful in your research, please cite:

```
@inproceedings{wu2021spatialtemporal,
      title={Cross-attention Spatio-temporal Context Transformer for Semantic Segmentation of Historical Maps}, 
      author={**Sidi Wu** and **Yizi Chen** and Konrad Schindler and Lorenz Hurni},
      year={2023},
      <!-- eprint={2109.01605}, -->
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Sources documents


### Project Structure

Structure of this repository:

```
|
├── dataset                     <- Dataset for training
├── img                         <- Images
├── loss                        <- Loss function
├── model                       <- Jupyter notebooks and Python scripts.
│   ├── spa                     <- Spatial model
│   ├── spatem                  <- Spatial+temporal model
|   ├── tem                     <- Temporal model
│   ├── unet                    <- Unet model
│   ├── unet3d                  <- 3d unet model
│   ├── utae                    <- L-TAE model
├──viz                          <- Visualization utiliy function
├──data.py                      <- Dataloader
├──environment.yml              <- Conda environment .yml file
├──log.py                       <- Log management file
├──train_*.py                   <- Training codes for different models (* segformer, spatial_temporal, ...)
└── README.md
```

## Installation :star2:

### 1. Create and activate conda environment

```
conda env create -f environment.yml
conda activate sigspatial
```

### 2. Download spatial-temporal historical maps dataset

** The dataset will be released soon...

## How to use :rocket:

### 1. Train models

```
# Segformer
python train_segformer.py --cuda --gpu 0 --lr 5e-4 --batch-size 10

# U-spa-temp (with different head options)
python train_spatial_temporal.py --cuda --gpu 0 --lr 5e-4 --batch-size 10  --n_head 4
python train_spatial_temporal.py --cuda --gpu 0 --lr 5e-4 --batch-size 10  --n_head 8
python train_spatial_temporal.py --cuda --gpu 0 --lr 5e-4 --batch-size 10  --n_head 16

# U-spa
python train_spatial.py --cuda --gpu 0 --lr 5e-4 --batch-size 10

# U-temp
python train_temporal.py --cuda --gpu 0 --lr 5e-4 --batch-size 10

# Original U-Net
python train_unet.py --cuda --gpu 0 --lr 5e-4 --batch-size 10

# 3d-unet
python train_unet3d.py --cuda --gpu 0 --lr 5e-4 --batch-size 10

# U-TAE
python train_utae.py --cuda --gpu 0 --lr 5e-4 --batch-size 10
```

Results and weights are saved at `training_info/`.

### 2. Inference


### 3. Acknowledgement
We appreciate helps from:  
* public code [U-TAE](https://github.com/VSainteuf/utae-paps.git)
