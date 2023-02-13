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
### Feb 11, 2023
Added <span style="color:green;font-weight:700;font-size:16px"> **training logging** </span> using [MLflow](https://mlflow.org/)
<span style="color:red;font-weight:700;font-size:15px">
    **Example**:
</span>
 to check logged items, run
```
mlflow ui
```

### Feb 10, 2023
Added <span style="color:green;font-weight:700;font-size:16px"> [app.py](https://github.com/etetteh/low-code-ml-dl/blob/main/deep_learning/image_classification/app.py) </span> to run inference with [FastAPI](https://fastapi.tiangolo.com/) on a single or multiple images by passing a JSON file.\
**Sample JSON file**
```
{
  "onnx_model_path": "swinv2_cr_tiny_ns_224/best_model.onnx",
  "imgs_paths": [<path_to_image1>, <path_to_image2>, <path_to_image3>, <path_to_image4>],
  "dataset_dir_or_classes_file": <path_to_dataset_dir_or_classes_file>
}
```

<span style="color:red;font-weight:700;font-size:15px">**Example**:</span>
to predict the class and probability score of the images in the JSON file, run

```
uvicorn app:app --reload
```

**Sample output**:
```
[
  {
    "image 0": {
      "Predicted Label": "rain",
      "Probability": 43.74
    }
  },
  {
    "image 1": {
      "Predicted Label": "cloudy",
      "Probability": 43.97
    }
  },
  {
    "image 2": {
      "Predicted Label": "shine",
      "Probability": 92.02
    }
  },
  {
    "image 3": {
      "Predicted Label": "sunrise",
      "Probability": 87.66
    }
  }
]
```

### Feb 07, 2023
Added <span style="color:green;font-weight:700;font-size:16px"> [inference.py](https://github.com/etetteh/low-code-ml-dl/blob/main/deep_learning/image_classification/inference.py) </span> to run inference on a single or multiple images using the best model saved in [ONNX](https://onnx.ai/) format.\
<span style="color:red;font-weight:700;font-size:15px">
    **Example**:
</span>
 to predict the class and probability score of an image using a `swinv2_cr_tiny_ns_224` model, run
```
python inference.py \
    --onnx_model_path  swinv2_cr_tiny_ns_224/best_model.onnx \
    --imgs_paths <paths_to_images> \
    --dataset_dir_or_classes_file <path_to_dataset_dir_or_classes_file>
```
Note that `dataset_dir_or_classes_file` takes as argument your dataset directory or a text file containing the classes 
### Jan 29, 2023
Added <span style="color:green;font-weight:700;font-size:16px"> [tune.py](https://github.com/etetteh/low-code-ml-dl/blob/main/deep_learning/image_classification/tune.py) </span> for hyperparameter tuning functionality using [Ray Tune](https://www.ray.io/ray-tune).\
<span style="color:red;font-weight:700;font-size:15px">
    **Example**:
</span> to tune the batch size, learning rate and weight decay (passing tune_opt by default tunes the learning rate and weight decay, and also momentum in the case of using SGD optimizer) using population based training algorithm and a `swinv2_cr_tiny_ns_224` model, run:
```
python tune.py \
    --name <experiment name> \
    --output_dir <model_checkpoint_dir> \
    --dataset_dir <dataset_dir> \
    --model_name swinv2_cr_tiny_ns_224 \
    --tune_batch_size \
    --tune_opt \
    --pbt 
```

### Jan 25, 2023
Added <span style="color:green;font-weight:700;font-size:16px"> [explainability.py](https://github.com/etetteh/low-code-ml-dl/blob/main/deep_learning/image_classification/explainability.py) </span> for model explainability functionality using [SHAP](https://shap.readthedocs.io/en/latest/index.html#). You can now understand the decision or prediction made by the best performing model.\
<span style="color:red;font-weight:700;font-size:15px">
    **Example**:
</span> to explain the performance of a `swinv2_cr_tiny_ns_224` model on 4 samples of the validation data, run the following in a notebook

```
from explainability import explain_model

class Args:
    def __init__(self):
        self.output_dir = "<model_checkpoint_dir>"
        self.dataset_dir = "<dataset_dir>"
        self.model_name = "swinv2_cr_tiny_ns_224"
        self.crop_size = 224
        self.batch_size = 30
        self.num_workers = 8
        self.n_samples = 4
        self.max_evals = 1000
        self.topk = 4
        self.dropout = 0.2
    
args = Args()

explain_model(
    args = args
)
```

### Jan 17, 2023
The train script uses models from the [timm](https://github.com/rwightman/pytorch-image-models) library.
1. Specify any model you'd like to use. You can pass a single model name or a list of model names.\
<span style="color:red;font-weight:700;font-size:15px">
  Example:
</span> to train with the models `swinv2_cr_tiny_ns_224` and `vit_large_patch14_clip_224`, run
```
python train.py \
    --model_name swinv2_cr_tiny_ns_224 vit_large_patch14_clip_224
```
Please, ensure that when passing a list of models, all the models should have been trained on the same image size.

2. Train all models with specific size and specific image size. This is useful for model selection.\
<span style="color:red;font-weight:700;font-size:15px">
    Example:
</span> to finetune all `tiny` models from the `timm` library that were pre-trained with image size 224, run

```
python train.py \ 
    --model_size tiny \
    --crop_size 224
```

You can also pass `nano`, `small`, `base`, `large` or `giant` to train all models with that respective size

Run `python train.py --help` to see all the arguments you can pass during training. 


The training script:
* Checkpoints the model, which can be used to resume training
* Saves the best model weights, which can be used for inference or deployment
* Plots a confusion matrix and ROC curve of the best validation metrics

All results are on the validation dataset, and are saved in `output_dir/<model_name>`.

<span style="color:red;font-weight:700;font-size:15px">
    Example:
</span> Sample output on the validation set after running the following code is shown below

```
python train.py \
    --dataset_dir weather_data \ 
    --model_size tiny \
    --crop_size 224 \ 
    --output_dir sample_run0
```
**Sample Results**:
```
                                         model  accuracy     auc      f1  recall  precision
0           vit_tiny_r_s16_p8_224.augreg_in21k    1.0000  0.9998  1.0000  1.0000     1.0000
1                        swinv2_cr_tiny_ns_224    1.0000  0.9999  1.0000  1.0000     1.0000
2                 swin_tiny_patch4_window7_224    0.9917  0.9988  0.9909  0.9917     0.9904
3                             swin_s3_tiny_224    0.9817  0.9970  0.9817  0.9817     0.9817
4                     xcit_tiny_12_p8_224_dist    0.9762  0.9965  0.9746  0.9762     0.9743
5   vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k    0.9750  0.9983  0.9737  0.9750     0.9745
6            vit_tiny_patch16_224.augreg_in21k    0.9750  0.9988  0.9737  0.9750     0.9745
7              deit_tiny_distilled_patch16_224    0.9762  0.9968  0.9736  0.9762     0.9732
8    vit_tiny_patch16_224.augreg_in21k_ft_in1k    0.9717  0.9963  0.9724  0.9717     0.9735
9                         xcit_tiny_24_p16_224    0.9690  0.9949  0.9664  0.9690     0.9648
10                       deit_tiny_patch16_224    0.9679  0.9960  0.9660  0.9679     0.9654
11                    xcit_tiny_24_p8_224_dist    0.9650  0.9959  0.9658  0.9650     0.9676
12                     maxvit_tiny_tf_224.in1k    0.9674  0.9953  0.9651  0.9674     0.9639
13                   xcit_tiny_24_p16_224_dist    0.9662  0.9960  0.9650  0.9662     0.9641
14                        xcit_tiny_12_p16_224    0.9579  0.9920  0.9570  0.9579     0.9564
15                         xcit_tiny_12_p8_224    0.9579  0.9970  0.9570  0.9579     0.9564
16                         xcit_tiny_24_p8_224    0.9467  0.9931  0.9486  0.9467     0.9520
17                   xcit_tiny_12_p16_224_dist    0.9290  0.9857  0.9295  0.9290     0.9303

```

Model explanation\
<img src="https://github.com/etetteh/low-code-ml-dl/blob/main/deep_learning/image_classification/plots/model_explainability.png" height="550" width="900">

Confusion Matrix\
<img src="https://github.com/etetteh/low-code-ml-dl/blob/main/deep_learning/image_classification/plots/confusion_matrix.png" height="550" width="900">

ROC Curve\
<img src="https://github.com/etetteh/low-code-ml-dl/blob/main/deep_learning/image_classification/plots/roc_curve.png" height="550" width="900">


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
3. Pre-release version of [timm](https://github.com/rwightman/pytorch-image-models) for the models used in this research project
4. [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/) for computing metrics
5. [SHAP](https://shap.readthedocs.io/en/latest/index.html#) for model explainability
6. [ONNX](https://onnx.ai/) for exporting model for inference
7. [FastAPI](https://fastapi.tiangolo.com/) for running inference on single or multiple images 
8. [MLflow](https://mlflow.org/) for logging training 

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
