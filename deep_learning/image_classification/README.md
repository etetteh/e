# State-Of-The-Art Binary and Multi-class Image Classification

## Table of Contents

* [Getting Started](getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Usage](#usage)
* [Documentation](#documentation)
* [License](#license)
* [Citation](#citation)

## Getting Started
The goal of this project is to provide a simple but efficient approach to image classification research by leveraging SOTA image models

## Prerequisites
1. Knowledge of Python programming to understand the code
2. Knowledge of machine learning and image classification
3. Knowledge to interpret metrics or results

## Installation
The script makes use of the following libraries, which can be installed following their respective instructions:
1. Python 3.10 or earlier. I recommend installing through [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
2. Latest stable release of [Pytorch](https://pytorch.org/get-started/locally/). Earlier versions should be okay.
3. Pre release version of (timm)[https://github.com/rwightman/pytorch-image-models]. Run 'pip install --pre timm' to install.
4. (TorchMetrics)[https://torchmetrics.readthedocs.io/en/stable/]

## Usage
The script uses models from the `timm` library.
1. Specify any model you'd like to use. You can pass a single model name or a list of model names. Example:
`python train.py --model vit_base_patch16_clip_384.laion2b_ft_in1k maxvit_base_tf_512.in1k
Please, ensure that when passing a list of models, all the models should have been trained on the same image size.

2. Train all models with specific size and specific image size. This is useful for model selection. Example, to finetune all `tiny` models from the `timm` library that was pre-trained with image size 224, run
```
python train.py --model_size tiny --crop_size 224
```
You can also pass `small`, `base`, `large` or `giant` to train all models with that respective size

3. Run `python train.py --help` to see all the arguments you can pass

## Documentation
Comming soon!

## License

## Citation
@misc{etetteh2023,
  author = {Enoch Tetteh},
  title = {Low Code Machine Learning And Deep Learning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {},
  howpublished = {\url{https://github.com/etetteh/low-code-ml-dl}}
} 