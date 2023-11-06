# Low Code Image Classification Inference Tutorial

Welcome to the Low Code Image Classification Inference Tutorial. In this tutorial, we will guide you through the process of evaluating a trained image classification model with minimal code. We will be using the best model saved in ONNX format for inference.

<!-- <video width="640" height="360" controls>
  <source src="URL_TO_VIDEO.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video> -->

## Getting Started

Let's start by setting up the environment. If you haven't already, clone the repository where the project is hosted. If you've already cloned it, you can update it with a git pull:

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

Before we proceed, make sure you have the required packages and libraries installed. You can install them using the following command:

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

### Inference on Test Data

Now, let's run inference on the test data using a trained model. We use the same command during training with the only exception being the added `--test_only` argument. Remember that we assume you have trained a model already following the previous tutorial on Model Training.  

**Sample Command For Model Inference On Test data**:

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
        --lr 0.001 \
        --test_only
```

**Expected output results after inferencing:**

```bash
Running evaluation on test data with the best model: efficientvit_b1
Val Metrics - loss: 0.0490 | accuracy: 0.9510 | auc: 0.9961 | f1: 0.9535 | recall: 0.9510 | precision: 0.9573 | Confusion Matrix 
[[28  0  2  0]
 [ 2 20  0  0]
 [ 1  0 25  0]
 [ 0  0  0 37]]
```

### Inference on a Single Image

You can also perform inference on a single image using a specific model. Example usage:

**Sample Command For Model Inference On Single Image**:

```python
>>> !accelerate launch \
        inference.py \
        --onnx_model_path enext/efficientvit_b1/best_model.onnx \
        --img_path weather_data/test/cloudy/cloudy205.jpg \
        --dataset_dir_or_classes_file weather_data \
        --output_dir enext/infer_results
```

Output after running the inference.

```bash
Inference results have been saved to enext/infer_results/inference_results.json
```

**View the results**:

```python
>>> !cat enext/infer_results/inference_results.json
{
    "cloudy205.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ]
}
```

### Inference on Multiple Images

To run inference on multiple images in a directory, use the following command. Example usage:

```python
>>> !accelerate launch \
        inference.py \
        --onnx_model_path enext/efficientvit_b1/best_model.onnx \
        --img_path weather_data/test/cloudy \
        --dataset_dir_or_classes_file weather_data \
        --output_dir enext/infer_multiple_results
```

Output after running the inference.

```bash
Inference results have been saved to enext/infer_multiple_results/inference_results.json
```

```python
>>> !cat enext/infer_multiple_results/inference_results.json
{
    "cloudy233.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy83.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy197.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 0.99
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        }
    ],
    "cloudy185.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 0.99
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy193.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy206.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy174.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy262.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        }
    ],
    "cloudy35.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy81.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        }
    ],
    "cloudy106.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 0.99
        },
        {
            "Predicted class": "shine",
            "Probability": 0.01
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy158.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy294.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy163.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy235.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        }
    ],
    "cloudy25.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 0.9
        },
        {
            "Predicted class": "shine",
            "Probability": 0.09
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy287.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy84.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy205.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy111.jpg": [
        {
            "Predicted class": "shine",
            "Probability": 0.81
        },
        {
            "Predicted class": "cloudy",
            "Probability": 0.19
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy159.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        }
    ],
    "cloudy56.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        }
    ],
    "cloudy291.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 0.99
        },
        {
            "Predicted class": "shine",
            "Probability": 0.01
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy135.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 0.97
        },
        {
            "Predicted class": "shine",
            "Probability": 0.03
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy115.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy137.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 0.99
        },
        {
            "Predicted class": "rain",
            "Probability": 0.01
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy90.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "rain",
            "Probability": 0.0
        }
    ],
    "cloudy114.jpg": [
        {
            "Predicted class": "shine",
            "Probability": 0.63
        },
        {
            "Predicted class": "cloudy",
            "Probability": 0.37
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy150.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ],
    "cloudy122.jpg": [
        {
            "Predicted class": "cloudy",
            "Probability": 1.0
        },
        {
            "Predicted class": "shine",
            "Probability": 0.0
        },
        {
            "Predicted class": "sunrise",
            "Probability": 0.0
        }
    ]
}
```

## Additional Configuration Options

To further customize your model inferencing process, you can explore additional configuration options by running:

```python
>>> !python inference.py --help
```

With these commands, you can efficiently evaluate your trained models or perform inference on image classification tasks. Enjoy the process of inference with minimal code!
