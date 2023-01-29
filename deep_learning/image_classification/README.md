# State-Of-The-Art Binary and Multi-class Image Classification

## Table of Contents

* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Usage](#usage)
* [Documentation](#documentation)
* [License](#license)
* [Citation](#citation)

## Usage
### Jan 29, 2023
Added **hyperparameter tuning** functionality using [Ray Tune](https://www.ray.io/ray-tune).\
Example: to tune the batch size, learning rate and weight decay (passing tune_opt by default tunes the learning rate and weight decay, and also momentum in the case of using SGD optimizer) using population based training algorithm, run:
```
python tune.py \
    --name <experiment name> \
    --output_dir <model_checkpoint_dir> \
    --dataset_dir <dataset_dir> \
    --model <name of model> \
    --tune_batch_size \
    --tune_opt \
    --pbt 
```

### Jan 25, 2023
Added **model explainability** functionality using [SHAP](https://shap.readthedocs.io/en/latest/index.html#). You can now understand the decision or prediction made by the best performing model.

Example: to explain the performance of `xcit_nano_12_p16_224_dist` on 4 samples of the validation data, run the following in a notebook
```
from explainability import explain_model

class Args:
    def __init__(self):
        self.output_dir = "<model_checkpoint_dir>"
        self.dataset_dir = "<dataset_dir>"
        self.model = "xcit_nano_12_p16_224_dist"
        self.crop_size = 224
        self.batch_size = 30
        self.num_workers = 8
        self.n_samples = 4
        self.max_evals = 1000
        self.topk = 4
    
args = Args()

explain_model(
    args = args
)

```

The script uses models from the `timm` library.
1. Specify any model you'd like to use. You can pass a single model name or a list of model names. Example:
```
python train.py \
--model beit_large_patch16_224 vit_large_patch14_clip_224
```
Please, ensure that when passing a list of models, all the models should have been trained on the same image size.

2Train all models with specific size and specific image size. This is useful for model selection. Example, to finetune all `tiny` models from the `timm` library that were pre-trained with image size 224, run
```
python train.py \ 
    --model_size tiny \
    --crop_size 224
```
You can also pass `nano`, `small`, `base`, `large` or `giant` to train all models with that respective size

Run `python train.py --help` to see all the arguments you can pass during training

**Results**\
The training script:
* Checkpoints the model, which can be used to resume training
* Saves the best model weights, which can be used for inference or deployment
* Plots a confusion matrix and ROC curve of the best validation metrics

All results are on the validation dataset, and are saved in the `output_dir` passed during training.

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
3. Pre release version of [timm](https://github.com/rwightman/pytorch-image-models). Run 'pip install --pre timm' to install.
4. [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/) for computing metrics
5. [SHAP](https://shap.readthedocs.io/en/latest/index.html#) for model explainability

## Documentation
Coming soon!

## License

## Citation
```
@misc{etetteh2023,
  author = {Enoch Tetteh},
  title = {Low Code Machine Learning And Deep Learning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {},
  howpublished = {\url{https://github.com/etetteh/low-code-ml-dl}}
} 
```
