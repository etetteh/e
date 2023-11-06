# Low Code Image Classification Training Tutorial

Welcome to the Low Code Image Classification Training Tutorial. In this tutorial, we will guide you through the process of training an image classification model with minimal code, using a variety of model size and architecture options.

<!-- <video width="640" height="360" controls>
  <source src="URL_TO_VIDEO.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video> -->

## Getting Started

First, let's clone the repository where the project is hosted. If you already have it, we'll perform a git pull to update it:

```python
>>> import os

>>> # Define variables for the repository URL and directory.
>>> REPO_URL = "https://github.com/etetteh/e.git"
>>> REPO_DIR = "e"

>>> # Check if the repository directory already exists and update it, otherwise clone it
>>> if os.path.exists(REPO_DIR):
>>>     print("Repository already exists. Performing git pull...")
>>>     %cd $REPO_DIR
>>>     !git pull
>>> else:
>>>     print("Cloning repository...")
>>>     !git clone $REPO_URL $REPO_DIR
```

Now, navigate to the image classification directory:

```python
>>> os.chdir("e/deepmindflow/image_classification/")
```

## Prerequisites

Before we begin, make sure you have the required packages and libraries installed. You can use the following command to install them:

```python
>>> !pip install -r requirements.txt
```

## Prepare the Sample Dataset

In this tutorial, we'll use a sample dataset. You can download and extract it with the following commands:

```python
>>> !wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/4drtyfjtfy-1.zip -qq
>>> !unzip -q 4drtyfjtfy-1.zip
>>> !unzip -q dataset2.zip
```

Next, we'll use Python code to process the dataset. We have created a utility module to assist with this:

```python
>>> import utils

>>> # Create image subclasses and save them in a `weather` directory
>>> data_process = utils.CreateImgSubclasses("dataset2", "weather")

>>> # Get image class names
>>> classes = data_process.get_image_classes()
>>> print(sorted(classes))
['cloudy', 'rain', 'shine', 'sunrise']
```

Now split the dataset into train, validation and test sets.

```python
>>> # Create subdirectories for each image class
>>> class_dirs = data_process.create_class_dirs(classes)

>>> # Copy images to their respective subdirectories
>>> weather_data = data_process.copy_images_to_dirs()

>>> # Create train, validation, and test dataset splits
>>> utils.create_train_val_test_splits(img_src="weather", img_dest="weather_data", ratio=(0.8, 0.10, 0.1))
```

## Cleaning Up

After dataset processing, you can remove unnecessary data:

```python
>>> !rm -rf weather 4drtyfjtfy-1.zip* dataset2*
```

```python
>>> from PIL import Image
>>> img = Image.open("weather_data/train/cloudy/cloudy1.jpg")
>>> img
```

Here is a sample cloudy image:

![sample cloudy image](../sample_cloudy_image.png)

## Model Training

Now, we'll train our image classification model. You have three options based on the model's size, module, or name:

### Size-Based Model Training

**Description**:

This option is ideal when you want to train your image classification model with a specific model size (in this case, "tiny"). Model size can significantly affect training speed and memory consumption. These are the available options to choose from: `"nano", "tiny", "small", "base", "large", and "giant"`.

**Display some available `tiny` models:**

```python
>>> import utils
>>> from argparse import Namespace 

>>> # Create a namespace object 'args' with specific configuration settings.
>>> args = Namespace(crop_size=224, model_size="tiny", module=None)

>>> # Retrieve a list of model names that match the given configuration.
>>> models = utils.get_matching_model_names(args)

>>> # Print the list of matching model names.
>>> print(*models, sep='\n')
deit_tiny_patch16_224.fb_in1k
eva02_tiny_patch14_224.mim_in22k
maxvit_tiny_rw_224.sw_in1k
maxvit_tiny_tf_224.in1k
swin_tiny_patch4_window7_224.ms_in1k
swin_tiny_patch4_window7_224.ms_in22k
swin_tiny_patch4_window7_224.ms_in22k_ft_in1k
swinv2_cr_tiny_ns_224.sw_in1k
tiny_vit_5m_224.dist_in22k
tiny_vit_5m_224.dist_in22k_ft_in1k
tiny_vit_5m_224.in1k
tiny_vit_11m_224.dist_in22k
tiny_vit_11m_224.dist_in22k_ft_in1k
tiny_vit_11m_224.in1k
tiny_vit_21m_224.dist_in22k
tiny_vit_21m_224.dist_in22k_ft_in1k
tiny_vit_21m_224.in1k
vit_tiny_patch16_224.augreg_in21k
vit_tiny_patch16_224.augreg_in21k_ft_in1k
vit_tiny_r_s16_p8_224.augreg_in21k
vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k
xcit_tiny_12_p8_224.fb_dist_in1k
xcit_tiny_12_p8_224.fb_in1k
xcit_tiny_12_p16_224.fb_dist_in1k
xcit_tiny_12_p16_224.fb_in1k
xcit_tiny_24_p8_224.fb_dist_in1k
xcit_tiny_24_p8_224.fb_in1k
xcit_tiny_24_p16_224.fb_dist_in1k
xcit_tiny_24_p16_224.fb_in1k
```

**Sample Command For Training**:

```python
>>> !accelerate launch \
        --mixed_precision=fp16 \
        --gradient_accumulation_steps=2 \
        --gradient_clipping=1 \
        train.py \
        --dataset weather_data \
        --output_dir tiny \
        --experiment_name tiny \
        --model_size tiny \
        --prune \
        --feat_extract \
        --sched_name one_cycle \
        --opt_name lion \
        --epochs 3 \
        --batch_size 32 \
        --lr 0.001
```

**Expected output results after training**:

```bash
Model performance against other models:
                                            model     acc     auc      f1  precision  recall
0                      maxvit_tiny_rw_224.sw_in1k  0.9362  0.9905  0.9382     0.9413  0.9362
1             tiny_vit_21m_224.dist_in22k_ft_in1k  0.9312  0.9864  0.9297     0.9344  0.9312
2                     tiny_vit_21m_224.dist_in22k  0.9186  0.9846  0.9173     0.9237  0.9186
3                           tiny_vit_11m_224.in1k  0.9157  0.9831  0.9130     0.9148  0.9157
4               xcit_tiny_12_p16_224.fb_dist_in1k  0.9081  0.9777  0.9022     0.9003  0.9081
5                           tiny_vit_21m_224.in1k  0.8826  0.9823  0.8827     0.8901  0.8826
6               xcit_tiny_24_p16_224.fb_dist_in1k  0.8836  0.9804  0.8813     0.8805  0.8836
7            swin_tiny_patch4_window7_224.ms_in1k  0.8764  0.9829  0.8767     0.8780  0.8764
8           swin_tiny_patch4_window7_224.ms_in22k  0.8686  0.9882  0.8601     0.8846  0.8686
9                     xcit_tiny_24_p8_224.fb_in1k  0.8424  0.9694  0.8506     0.8673  0.8424
10               eva02_tiny_patch14_224.mim_in22k  0.8464  0.9856  0.8463     0.8679  0.8464
11                  swinv2_cr_tiny_ns_224.sw_in1k  0.8529  0.9783  0.8459     0.8705  0.8529
12                    xcit_tiny_12_p8_224.fb_in1k  0.8376  0.9715  0.8329     0.8303  0.8376
13               xcit_tiny_12_p8_224.fb_dist_in1k  0.8260  0.9667  0.8269     0.8353  0.8260
14               xcit_tiny_24_p8_224.fb_dist_in1k  0.8221  0.9614  0.8217     0.8251  0.8221
15                    tiny_vit_11m_224.dist_in22k  0.8088  0.9501  0.8166     0.8498  0.8088
16  swin_tiny_patch4_window7_224.ms_in22k_ft_in1k  0.8262  0.9780  0.8118     0.8709  0.8262
17            tiny_vit_11m_224.dist_in22k_ft_in1k  0.8057  0.9538  0.8037     0.8551  0.8057
18                           tiny_vit_5m_224.in1k  0.8286  0.9837  0.7927     0.8507  0.8286
19                        maxvit_tiny_tf_224.in1k  0.7810  0.9567  0.7826     0.8208  0.7810
20     vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k  0.7800  0.9746  0.7706     0.8625  0.7800
21                   xcit_tiny_24_p16_224.fb_in1k  0.7617  0.9412  0.7673     0.7881  0.7617
22                  deit_tiny_patch16_224.fb_in1k  0.7669  0.9303  0.7592     0.8151  0.7669
23                   xcit_tiny_12_p16_224.fb_in1k  0.7643  0.9456  0.7579     0.7785  0.7643
24             tiny_vit_5m_224.dist_in22k_ft_in1k  0.7726  0.9178  0.7549     0.7626  0.7726
25             vit_tiny_r_s16_p8_224.augreg_in21k  0.7314  0.9691  0.7436     0.8328  0.7314
26                     tiny_vit_5m_224.dist_in22k  0.7026  0.9359  0.7032     0.7529  0.7026
27      vit_tiny_patch16_224.augreg_in21k_ft_in1k  0.6793  0.9724  0.5957     0.7773  0.6793
28              vit_tiny_patch16_224.augreg_in21k  0.5217  0.9035  0.4771     0.7820  0.5217

ONNX model saved to tiny/maxvit_tiny_rw_224.sw_in1k/best_model.onnx
Exported best performing model, maxvit_tiny_rw_224.sw_in1k, to ONNX format. File is located in tiny/maxvit_tiny_rw_224.sw_in1k
All results have been saved at e/deepmindflow/image_classification/tiny
```

### Module-Based Model Training

**Description**:

Module-based training allows you to choose a specific neural network module or architecture (e.g., "edgenext") for your image classification model.
Before we dive into the training process, let's take a look at the available modules or architectures that you can use as the backbone of your image classification model. These modules offer different approaches and capabilities, allowing you to choose the one that best suits your project's requirements and performance goals. Here are few examples of modules: `"beit", "convnext", "deit", "resnet", "vision_transformer", "efficientnet", "xcit", "regnet", "nfnet", "metaformer", "fastvit", "efficientvit_msra"` and more.
Run the following command to get the names of all the modules for your use `timm.list_modules()`.

**Display available models with the `edgenext` module:**

```python
>>> import utils
>>> from argparse import Namespace 

>>> # Create a namespace object 'args' with specific configuration settings.
>>> args = Namespace(crop_size=224, module="edgenext")

>>> # Retrieve a list of model names that match the given configuration.
>>> models = utils.get_matching_model_names(args)

>>> # Print the list of matching model names.
>>> print(*models, sep='\n')
edgenext_base.in21k_ft_in1k
edgenext_base.usi_in1k
edgenext_small.usi_in1k
edgenext_small_rw.sw_in1k
edgenext_x_small.in1k
edgenext_xx_small.in1k
```

**Sample Command For Training**:

```python
>>> !accelerate launch \
        --mixed_precision=fp16 \
        --gradient_accumulation_steps=2 \
        --gradient_clipping=1 \
        train.py \
        --dataset weather_data \
        --output_dir enext \
        --experiment_name enext \
        --module edgenext \
        --prune \
        --feat_extract \
        --sched_name one_cycle \
        --opt_name lion \
        --epochs 10 \
        --batch_size 32 \
        --lr 0.001
```

**Expected output results after training**:

```bash
Model performance against other models:
                         model     acc     auc      f1  precision  recall
0       edgenext_base.usi_in1k  0.9407  0.9871  0.9394     0.9389  0.9407
1  edgenext_base.in21k_ft_in1k  0.9131  0.9881  0.9114     0.9242  0.9131
2        edgenext_x_small.in1k  0.8955  0.9811  0.8943     0.9065  0.8955
3       edgenext_xx_small.in1k  0.8814  0.9790  0.8844     0.8894  0.8814
4    edgenext_small_rw.sw_in1k  0.8564  0.9739  0.8507     0.8710  0.8564
5      edgenext_small.usi_in1k  0.8545  0.9761  0.8437     0.8529  0.8545

ONNX model saved to enext/edgenext_base.usi_in1k/best_model.onnx
Exported best performing model, edgenext_base.usi_in1k, to ONNX format. File is located in enext/edgenext_base.usi_in1k
All results have been saved at /content/e/deepmindflow/image_classification/enext
```

## Name-Based Model Training

**Description**:

Name-based model training offers the flexibility to choose from a list of pre-defined model names, such as "efficientvit_b1", "fastvit_sa12", "mobileone_s3", "ghostnetv2_160", and "repghostnet_200". This option is useful when you want to experiment with different architectures or train a specific model. Run the following command to get all the model names for your use `timm.list_models(pretrained=True)`.

**Sample Command For Training**:

```python
>>> !accelerate launch \
        --mixed_precision=fp16 \
        --gradient_accumulation_steps=2 \
        --gradient_clipping=1 \
        train.py \
        --dataset weather_data \
        --output_dir enext \
        --experiment_name enext \
        --model_name efficientvit_b1 fastvit_sa12 mobileone_s3 ghostnetv2_160 repghostnet_200 \
        --prune \
        --feat_extract \
        --sched_name one_cycle \
        --opt_name lion \
        --epochs 10 \
        --batch_size 32 \
        --lr 0.001
```

**Expected output results after training**:

```bash
Model performance against other models:
             model     acc     auc      f1  precision  recall
0     mobileone_s3  0.9733  0.9957  0.9732     0.9738  0.9733
1  repghostnet_200  0.9579  0.9879  0.9547     0.9537  0.9579
2  efficientvit_b1  0.9360  0.9898  0.9367     0.9386  0.9360
3     fastvit_sa12  0.9295  0.9851  0.9269     0.9280  0.9295
4   ghostnetv2_160  0.9310  0.9906  0.9244     0.9256  0.9310

ONNX model saved to enext/mobileone_s3/best_model.onnx
Exported best performing model, mobileone_s3, to ONNX format. File is located in enext/mobileone_s3
All results have been saved at /content/e/deepmindflow/image_classification/enext
```

## Additional Configuration Options

To further customize your training process, you can explore additional configuration options by running:

```python
>>> !python train.py --help
```

This will provide you with a comprehensive list of available options to fine-tune your image classification model according to your specific needs and constraints.

With these training options, you have the flexibility to experiment with different model sizes, modules, and architectures, enabling you to achieve the best results for your image classification task. Happy training!
