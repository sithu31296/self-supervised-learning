# Self Supervised Image Classification

## Introduction

Read a blog post from FAIR >> [Self-supervised learning: The dark matter of intelligence](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/).

<!--- 
https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html

https://amitness.com/2020/02/illustrated-self-supervised-learning/

https://www.fast.ai/2020/01/13/self_supervised/

https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html

https://vimeo.com/390347111
--->


## Features

Datasets
* [ImageNet](https://image-net.org/)

Models
* [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681v2)
* [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294v2)

## Model Comparison

Method | Model | ImageNet Top1 Acc (Linear) | ImageNet Top1 Acc (Linear) | Params (M) | Weights
--- | --- | --- | --- | --- | ---
DINO | XCiT-M24/8 | 80.3 | 77.9 | 84 | [model](https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth)/[checkpoint](https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_full_checkpoint.pth)
DINO | XCiT-S12/8 | 79.2 | 77.1 | 26 | [model](https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth)/[checkpoint](https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_full_checkpoint.pth)
DINO | ViT-B/8 | 80.1 | 77.4 | 85 | [model](https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth)/[checkpoint](https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth)
DINO | ViT-S/8 | 79.7 | 78.3 | 21 | [model](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth)/[checkpoint](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth)

## Configuration 

Create a configuration file in `configs`. Sample configuration for ImageNet dataset with DINO can be found [here](configs/dino.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

## Training

### Single GPU
```bash
$ python tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

### Multiple GPUs

Traing with 2 GPUs:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Evaluation

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

### Linear Classification

This will train a supervised linear classifier on top of trained weights and evaluate the result.

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/val_linear.py --cfg configs/CONFIG_FILE_NAME.yaml
```

### k-NN Classification

```bash
$ python -m torch.distributed.launch --nproc_per_node=1 --use_env tools/val_knn.py --cfg configs/CONFIG_FILE_NAME.yaml
```


## Attention Visualization

Make sure to set `MODEL_PATH` of the configuration file to model's weights.

```bash
$ python tools/visualize_attention.py --cfg configs/CONFIG_FILE_NAME.yaml
```