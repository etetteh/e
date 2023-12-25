import os
import re
import json
import random
import shutil
from torch.distributions import Beta

from glob import glob
from pathlib import Path
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datasets import Dataset
import multiprocessing

import timm
import torch
import datasets
import numpy as np
import splitfolders
import torch.nn.utils.prune as prune
import torch.optim.lr_scheduler as lr_scheduler

from accelerate import Accelerator
from datasets import load_dataset
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as f
from timm.optim import create_optimizer_v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

torch.jit.enable_onednn_fusion(True)


def set_seed_for_worker(worker_id: Optional[int]) -> Union[int, None]:
    """
    Sets the seed for NumPy and Python's random module for the given worker.
    If no worker ID is provided, uses the initial seed for PyTorch and returns "<initial_seed_value>" as a string.

    Parameters:
        worker_id (Optional[int]): The ID of the worker. If None, uses the initial seed.

    Returns:
        Union[int, None]: The seed used for the worker, or None if no worker ID was provided.

    Raises:
        ValueError: If the worker ID is not an integer.

    Examples:
        >>> worker_id = 1  # Example worker ID
        >>> seed = set_seed_for_worker(worker_id)
        >>> print(seed)  # Seed used for the worker
        1

        >>> seed = set_seed_for_worker(None)  # No worker ID provided
        >>> print(seed)  # Initial seed for PyTorch
        None
    """
    if worker_id is not None:
        if not isinstance(worker_id, int):
            raise ValueError("Worker ID must be an integer.")
        worker_seed = worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        return worker_seed
    else:
        initial_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(initial_seed)
        random.seed(initial_seed)
        return None


def print_header(message: str, sep: str = "=") -> None:
    """
    Print a header line with a message surrounded by a separator.

    Parameters:
        message (str): The message to be included in the header.
        sep (str): Separator character(s) to be used (default is '=').

    Examples:
        >>> print_header("Welcome to the Chatbot")
        ======================
        Welcome to the Chatbot
        ======================

        >>> print_header("Instructions", sep='-')
        ------------
        Instructions
        ------------

        >>> print_header("Important Notice", sep='*')
        ****************
        Important Notice
        ****************

        Edge Cases:
        >>> print_header("", sep='#')  # Empty message
        #
        <BLANKLINE>
        #

        >>> print_header("Custom Separator", sep='##')  # Multi-character separator
        ################################
        Custom Separator
        ################################
    """
    line = sep * max(len(message), 1)  # Ensure there's at least one separator character
    header = f"{line}\n{message}\n{line}"
    print(header)


def get_model_run_id(run_ids: Dict[str, str], model_name: str = None) -> Optional[str]:
    """
    Get the run ID of a specific model from a dictionary of run IDs.

    Parameters:
        run_ids (Dict[str, str]): A dictionary mapping model names to run IDs.
        model_name (str, optional): The name of the model for which to retrieve the run ID.
                                    If not provided or the model name is not in the dictionary, returns None.

    Returns:
        Optional[str]: The run ID of the specified model, or None if the model is not in the dictionary.

    Examples:
        >>> run_ids = {"model1": "1234", "model2": "5678", "model3": "9012"}  # Example dictionary of run IDs

        >>> model_name = "model2"  # Example model name to retrieve the run ID
        >>> get_model_run_id(run_ids, model_name)
        '5678'

        >>> model_name = "model4"  # Non-existing model name
        >>> get_model_run_id(run_ids, model_name) is None
        True

        Edge Cases:
        >>> get_model_run_id(run_ids, None) is None  # No model name provided
        True

        >>> get_model_run_id({}, "model1") is None  # Empty dictionary
        True
    """
    return run_ids.get(model_name, None)


def write_dictionary_to_json(dictionary: Dict, file_path: str) -> None:
    """
    Write a dictionary object to a JSON file at the given file path.
    If the file already exists, its content will be overwritten.

    Parameters:
        dictionary (Dict): The dictionary object to be written to the JSON file.
        file_path (str): The path to the JSON file.

    Returns:
        None

    Raises:
        IOError: If there is an issue with opening or writing to the file.
            - doc: Raised when there is an issue with opening or writing to the file.

        json.JSONDecodeError: If there is an issue with JSON serialization.
            - doc: Raised when there is an issue with JSON serialization.
            - pos (int): The character position in the input string where the error occurred.

    Examples:
        >>> dictionary = {"key": "value"}  # Example dictionary object to be written
        >>> file_path = "data.json"  # Example file path
        >>> write_dictionary_to_json(dictionary, file_path)

        # To check the content of the data.json file, you can read it and print it:
        >>> with open(file_path, 'r') as file_in:
        ...     file_content = file_in.read()
        >>> print(file_content)
        {
            "key": "value"
        }

        # Example of handling IOError
        >>> try:
        ...     write_dictionary_to_json(dictionary, "/nonexistent_directory/data.json")
        ... except IOError as e:
        ...     print(f"An IOError occurred: {e}")
        An IOError occurred: Failed to write to JSON file '/nonexistent_directory/data.json': [Errno 2] No such file or directory: '/nonexistent_directory/data.json'

        # Example of handling JSONDecodeError
        # >>> invalid_dictionary = {"key": b"invalid_data"}
        # >>> try:
        # ...     write_dictionary_to_json(invalid_dictionary, "invalid.json")
        # ... except json.JSONDecodeError as e:
        # ...     print(f"JSONDecodeError occurred at character {e.pos}: {e}")
        # JSONDecodeError occurred at character 2: Expecting property name enclosed in double quotes: line 1 column 3 (char 2)
    """
    try:
        with open(file_path, "w") as file_out:
            json.dump(dictionary, file_out, indent=4)
    except (IOError, FileNotFoundError) as e:
        raise IOError(f"Failed to write to JSON file '{file_path}': {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to serialize dictionary to JSON: {e}", e.doc, e.pos) from e


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Reads a JSON file and returns its content as a dictionary.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The dictionary containing the content of the JSON file.

    Raises:
        IOError: If there is an issue with opening or reading the file.
        json.JSONDecodeError: If there is an issue with JSON parsing.
        RuntimeError: For any unexpected errors.

    Examples:
        >>> import os

        # Example JSON data and file path
        >>> json_data = {"key1": "value1", "key2": "value2"}
        >>> file_path = "data.json"

        # Creating a sample JSON file
        >>> with open(file_path, 'w') as json_file:
        ...     json.dump(json_data, json_file)

        # Reading the JSON file using read_json_file function
        >>> result = read_json_file(file_path)
        >>> expected_result = {"key1": "value1", "key2": "value2"}
        >>> assert result == expected_result

        # Clean up: Remove the created file
        >>> os.remove(file_path)
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except (IOError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read JSON file '{file_path}': {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def append_dictionary_to_json_file(new_dict: Dict[str, Any], file_path: str) -> None:
    """
    Append a dictionary to a JSON file at the given file path.
    If the file does not exist, it will be created.

    Parameters:
        new_dict (Dict[str, Any]): The dictionary to be appended to the JSON file.
        file_path (str): The path to the JSON file.

    Returns:
        None

    Raises:
        IOError: If there is an issue with opening or writing the file.
        json.JSONDecodeError: If there is an issue with JSON serialization.
        RuntimeError: For any unexpected errors.

    Examples:
        >>> import os

        # Example dictionary to be appended
        >>> new_dict = {"key": "value"}
        >>> file_path = "data.json"

        # Creating a sample JSON file with initial data
        >>> initial_data = {"existing_key": "existing_value"}
        >>> with open(file_path, 'w') as json_file:
        ...     json.dump(initial_data, json_file, indent=4)

        # Appending the new_dict to the JSON file using append_dictionary_to_json_file
        >>> append_dictionary_to_json_file(new_dict, file_path)

        # Expected Output:
        # The data.json file should now contain both dictionaries:
        # {
        #     "existing_key": "existing_value",
        #     "key": "value"
        # }

        # Clean up: Remove the created file
        >>> os.remove(file_path)
    """
    try:
        existing_data = {}
        if os.path.isfile(file_path):
            existing_data = read_json_file(file_path)

        existing_data.update(new_dict)

        with open(file_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)
    except (IOError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to write to JSON file '{file_path}': {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def read_json_lines_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read and parse a JSON Lines file from the given file path and return the data as a list of dictionaries.

    Parameters:
        file_path (str): The path to the JSON Lines file.

    Returns:
        List[Dict[str, Any]]: The data contained in the JSON Lines file as a list of dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
            - doc: Raised when the specified file does not exist.

        json.JSONDecodeError: If there is an issue with JSON decoding.
            - doc: Raised when there is an issue decoding JSON data.
            - pos (int): The character position in the input where the error occurred.

    Examples:
        >>> import pandas as pd

        # Create a sample DataFrame
        >>> data = [\
            {"label": "DRUG", "pattern": "aspirin"},\
            {"label": "DRUG", "pattern": "trazodone"},\
            {"label": "DRUG", "pattern": "citalopram"}\
        ]
        >>> df = pd.DataFrame(data)

        # Output the DataFrame in JSONL format
        >>> df.to_json("data.jsonl", orient="records", lines=True)

        >>> file_path = "data.jsonl"  # Example file path (created by Pandas)
        >>> data = read_json_lines_file(file_path)
        >>> print(data)
        [{'label': 'DRUG', 'pattern': 'aspirin'}, {'label': 'DRUG', 'pattern': 'trazodone'}, {'label': 'DRUG', 'pattern': 'citalopram'}]
    """
    try:
        with open(file_path, "r") as file_in:
            data = [json.loads(line) for line in file_in]
            return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {file_path} not found.") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON in {file_path}: {e}", e.doc, e.pos) from e


def keep_best_f1_score_files(directory: Path, num_files_to_keep: int):
    """
    Sorts files in the specified directory and keeps files with the best f1 scores.
    Files beyond the specified number of files to keep will be removed.

    Parameters:
        directory (str): The path to the directory containing the files.
        num_files_to_keep (int): The number of best f1 score files to keep.

    Examples:
        >>> import os
        >>> import random
        >>> from glob import glob
        >>> from pathlib import Path

        # Specify the directory and the number of files to create
        >>> directory_path = "resnet50_checkpoints"

        # Create the directory if it doesn't exist
        >>> directory_path = Path(directory_path)
        >>> if not directory_path.exists():
        ...     directory_path.mkdir(parents=True)

        # Predefined F1 scores for testing
        >>> f1_scores = [0.8971, 0.8632, 0.8819, 0.9854, 0.9531, 0.9342]

        # Create files with names following the format "best_model_{f1_score}.pth" in the specified directory.
        >>> for f1_score in f1_scores:
        ...     file_name = f"best_model_{f1_score:.4f}.pth"
        ...     file_path = directory_path / file_name
        ...     open(file_path, 'w').close()

        # Sort and keep the best files based on F1 score
        >>> keep_best_f1_score_files(directory_path, num_files_to_keep=2)

        >>> remaining_files = list(directory_path.glob("best_model_*.pth"))
        >>> print(remaining_files)
        [PosixPath('resnet50_checkpoints/best_model_0.9531.pth'), PosixPath('resnet50_checkpoints/best_model_0.9854.pth')]
    """
    directory_path = Path(directory)

    if not directory_path.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Sort files by F1 score in descending order
    file_list = list(directory_path.glob("best_model_*.pth"))
    sorted_files = sorted(file_list, key=lambda x: float(x.stem.split("_")[2]), reverse=True)

    # Remove files beyond the specified number to keep
    files_to_remove = sorted_files[num_files_to_keep:]
    [file_to_remove.unlink() for file_to_remove in files_to_remove]


def remove_items(items: List[str], to_remove: str | List[str] = None) -> List[str]:
    """Removes items from a list of strings that contain the specified string or strings.

    If `to_remove` is a string, it matches entire words. If it's a list, it matches any of the strings in the list.

    Args:
        items: A list of strings to be filtered.
        to_remove: The string or list of strings to be removed from the items.

    Returns:
        A filtered list of strings, where all items containing the specified string or strings have been removed.

    Raises:
        TypeError: If `items` is not a list of strings.

    Example:
        >>> items = ["eva02_base_patch14_224.mim_in22k",
        ...          "eva02_base_patch14_448.mim_in22k_ft_in1k",
        ...          "eva02_base_patch14_448.mim_in22k_ft_in22k",
        ...          "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        ...          "eva02_base_patch16_clip_224.merged2b",
        ...          "eva02_large_patch14_clip_224.merged2b",
        ...          "eva02_large_patch14_clip_336.merged2b"]
        >>> filtered_list = remove_items(items, "large")
        >>> print(filtered_list)
    """

    if not isinstance(items, list):
        raise TypeError("'items' argument must be a list of strings")

    if to_remove is None:
        return items

    if isinstance(to_remove, str):
        to_remove_pattern = re.escape(to_remove)
    elif isinstance(to_remove, list):
        to_remove_pattern = "|".join(re.escape(item) for item in to_remove)
    else:
        raise TypeError("'to_remove' argument must be a string or list of strings")

    filtered_list = [item for item in items if not re.search(to_remove_pattern, item)]

    return filtered_list


def load_image_dataset(args: Namespace) -> datasets.arrow_dataset.Dataset:
    """
    Load an image dataset using Hugging Face's 'datasets' library.

    Parameters:
        args (Namespace): Namespace containing the following attributes:
            - dataset (str or os.PathLike): The path to a local dataset directory or a Hugging Face dataset name
              (or an HTTPS URL for a remote dataset).
            - dataset_kwargs (str): The path to a JSON file containing kwargs of a Hugging Face dataset.

    Returns:
        datasets.arrow_dataset.Dataset: The loaded image dataset.

    Raises:
        ValueError: If the provided dataset is not a valid path to a directory, a string (Hugging Face dataset name),
                    or an HTTPS URL for a remote dataset.

    Examples:
        >>> args = Namespace(dataset="cifar10", dataset_kwargs="")
        >>> dataset = load_image_dataset(args)
        >>> print(dataset)
        DatasetDict({
            train: Dataset({
                features: ['image', 'labels'],
                num_rows: 40000
            })
            test: Dataset({
                features: ['image', 'labels'],
                num_rows: 10000
            })
            validation: Dataset({
                features: ['image', 'labels'],
                num_rows: 10000
            })
        })
    """
    if isinstance(args.dataset, Path) and args.dataset.is_dir():
        image_dataset = load_dataset("imagefolder", data_dir=str(args.dataset))
    elif isinstance(args.dataset, str):
        data_kwargs = {"path": args.dataset}
        if args.dataset_kwargs.endswith(".json"):
            with open(args.dataset_kwargs, 'r') as json_file:
                data_kwargs = json.load(json_file)
        image_dataset = load_dataset(**data_kwargs)
    else:
        raise ValueError("Dataset should be a path to a local dataset on disk, a dataset name of an image dataset "
                         "from Hugging Face datasets, or an HTTPS URL for a remote dataset.")

    image_column_names = image_dataset["train"].column_names
    if "img" in image_column_names:
        image_dataset = image_dataset.rename_column("img", "image")
    if "label" in image_column_names:
        image_dataset = image_dataset.rename_column("label", "labels")
    if "fine_label" in image_column_names:
        image_dataset = image_dataset.rename_column("fine_label", "labels")

    if "validation" not in image_dataset.keys() and "test" in image_dataset.keys():
        new = image_dataset["train"].train_test_split(test_size=0.2, stratify_by_column="labels")
        new["validation"] = new["test"]
        new["test"] = image_dataset["test"]
        image_dataset = new
    elif "validation" not in image_dataset.keys():
        new = image_dataset["train"].train_test_split(test_size=0.2, stratify_by_column="labels")
        new["validation"] = new["test"]
        new.pop("test")
        image_dataset = new
    elif "test" not in image_dataset.keys() and "validation" in image_dataset.keys():
        new = image_dataset["train"].train_test_split(test_size=0.2, stratify_by_column="labels")
        new["validation"] = image_dataset["validation"]
        image_dataset = new

    return image_dataset


def preprocess_train_eval_data(
        image_dataset: Dataset,
        data_transforms: Dict[str, callable]
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Preprocesses training and validation data in an image dataset using specified data transformations.

    This function applies the provided data transformations to the training and validation subsets
    of an image dataset. It is intended for preparing image data for training and evaluation in
    machine learning models.

    Parameters:
        image_dataset (Dict[str, Dataset]): A dictionary containing subsets of an image dataset, typically with keys
            "train" and "validation" pointing to dataset objects.
        data_transforms (Dict[str, callable]): A dictionary containing data transformation functions for training ("train")
            and validation ("val"). These functions should accept and process image data.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the preprocessed training and validation datasets.

    Raises:
        ValueError: If the provided data transformations do not match the keys "train" and "val" or
                    if the provided dataset subsets do not match the keys "train" and "validation".

    Examples:
        >>> from argparse import Namespace

        >>> args = Namespace( \
            dataset = "cifar10", \
            dataset_kwargs="", \
            crop_size=224, \
            val_resize=256,\
            interpolation="bilinear", \
            hflip=0.5, \
            aug_type="augmix", \
            mag_bins=30, \
            grayscale=False \
            )

        >>> data_transforms = get_data_augmentation(args)
        >>> dataset = load_image_dataset(args)

        >>> train_dataset, val_dataset, test_dataset = preprocess_train_eval_data(dataset, data_transforms)
        >>> train_dataset
        Dataset({
            features: ['image', 'labels'],
            num_rows: 40000
        })
    """
    if "train" not in image_dataset:
        raise ValueError("The 'train' dataset subset is missing.")
    if "validation" not in image_dataset:
        raise ValueError("The 'validation' dataset subset is missing.")
    if "test" not in image_dataset:
        raise ValueError("The 'test' dataset subset is missing.")
    if "train" not in data_transforms:
        raise ValueError("Data transformations for 'train' are missing.")
    if "val" not in data_transforms:
        raise ValueError("Data transformations for 'val' are missing.")

    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [
            data_transforms["train"](image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [
            data_transforms["val"](image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    train_dataset = image_dataset["train"].with_transform(preprocess_train)
    val_dataset = image_dataset["validation"].with_transform(preprocess_val)
    test_dataset = image_dataset["test"].with_transform(preprocess_val)

    return train_dataset, val_dataset, test_dataset


def apply_fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """
    Generate an adversarial example using the Fast Gradient Sign Method (FGSM).

    Parameters:
        image (torch.Tensor): The input image.
        epsilon (float): Perturbation magnitude for generating adversarial examples.
        data_grad (torch.Tensor): Gradient of the loss with respect to the image.

    Returns:
        torch.Tensor: Adversarial example perturbed using FGSM.

    Examples:
        >>> image = torch.rand(1, 3, 32, 32)  # A random image of size (1, 3, 32, 32)
        >>> epsilon = 0.05  # Perturbation magnitude
        >>> data_grad = torch.randn(1, 3, 32, 32)  # Gradient of loss w.r.t. image
        >>> adversarial_image = apply_fgsm_attack(image, epsilon, data_grad)
    """
    # Check if epsilon is a positive value
    if epsilon <= 0:
        raise ValueError("Epsilon must be a positive value.")

    # Check if the input tensors have compatible shapes
    if image.shape != data_grad.shape:
        raise ValueError("Input image and data_grad must have the same shape.")

    # Compute the sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Perturb the image using FGSM and clamp values to [0, 1]
    adversarial_image = image + epsilon * sign_data_grad
    adversarial_image = torch.clamp(adversarial_image, min=0, max=1)
    return adversarial_image


def apply_mixup(images: Tensor, targets: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    Applies Mixup augmentation to input data.

    This function performs Mixup augmentation, which blends input images and their corresponding
    targets based on a random mixing coefficient sampled from a Beta distribution.

    Parameters:
        images (Tensor): Input images as a tensor of shape (batch_size, channels, height, width).
        targets (Tensor): Corresponding targets as a tensor of shape (batch_size, num_classes).
        alpha (float, optional): Mixup parameter that controls the mixing strength.
            A smaller alpha value results in more conservative mixing. Defaults to 1.0.

    Returns:
        Tuple[Tensor, Tensor, Tensor, float]: A tuple containing the following elements:
            - Mixed images (Tensor): A tensor of mixed images with the same shape as 'images'.
            - Mixed labels (targets_a) (Tensor): A tensor of mixed labels with the same shape as 'targets'.
            - Original labels (targets_b) (Tensor): A tensor of original labels with the same shape as 'targets'.
            - Mixup factor (lambda) (float): The mixing coefficient used for the augmentation.

    Examples:
        >>> import torch
        >>> images = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        >>> mixed_images, mixed_labels, original_labels, lam = apply_mixup(images, targets, alpha=0.5)
        >>> mixed_images.shape
        torch.Size([2, 2])
        >>> 0.0 <= lam <= 1.0
        True
    """
    if alpha <= 0:
        raise ValueError("The 'alpha' parameter must be greater than zero.")

    if images.shape[0] != targets.shape[0]:
        raise ValueError("The number of images and targets must be the same.")

    beta_dist = Beta(alpha, alpha)
    lam = beta_dist.sample().item()

    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    mixed_images = lam * images + (1 - lam) * images[index, :]
    targets_a, targets_b = targets, targets[index]

    return mixed_images, targets_a, targets_b, lam


def apply_cutmix(images: torch.Tensor,
                 targets: torch.Tensor,
                 alpha: float = 1.0
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies CutMix augmentation to input data.

    CutMix is an augmentation technique that combines two images by replacing a part of one image with a part of another.

    Parameters:
        images (torch.Tensor): Input images as a tensor of shape (batch_size, channels, height, width).
        targets (torch.Tensor): Corresponding labels as a tensor of shape (batch_size, num_classes).
        alpha (float, optional): CutMix parameter that controls the mixing strength. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: A tuple containing the following elements:
            - Mixed images (torch.Tensor): A tensor of mixed images with the same shape as 'images'.
            - Mixed labels (targets_a) (torch.Tensor): A tensor of mixed labels with the same shape as 'targets'.
            - Original labels (targets_b) (torch.Tensor): A tensor of original labels with the same shape as 'targets'.
            - Cutmix factor (lambda) (float): The mixing coefficient used for the augmentation.

    Examples:
        >>> import torch
        >>> images = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        >>> targets = torch.tensor([[0.0, 1.0]])
        >>> mixed_images, mixed_labels, original_labels, lam = apply_cutmix(images, targets, alpha=0.5)
        >>> mixed_images.shape
        torch.Size([1, 1, 2, 2])
        >>> mixed_labels.shape
        torch.Size([1, 2])
        >>> original_labels.shape
        torch.Size([1, 2])
        >>> 0.0 <= lam <= 1.0
        True
    """
    batch_size, channels, height, width = images.size()
    index = torch.randperm(batch_size)
    lam = torch.tensor(np.random.beta(alpha, alpha))  # Convert lam to a PyTorch tensor
    lam = torch.max(lam, 1 - lam)  # Ensure lambda is always greater than 0.5

    # Generate random bounding box coordinates
    cut_ratio = torch.sqrt(1.0 - lam)
    cut_h = torch.tensor(height * cut_ratio, dtype=torch.int)
    cut_w = torch.tensor(width * cut_ratio, dtype=torch.int)
    cx = torch.randint(width, (1,)).item()
    cy = torch.randint(height, (1,)).item()
    bbx1 = torch.clamp(cx - cut_w // 2, 0, width)
    bby1 = torch.clamp(cy - cut_h // 2, 0, height)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, width)
    bby2 = torch.clamp(cy + cut_h // 2, 0, height)

    # Apply CutMix
    mixed_images = images.clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (width * height)

    # Adjust the labels
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam


def apply_normalization(args: Namespace, aug_list: List) -> List:
    """
    Apply normalization to the augmentation list based on grayscale conversion.

    Parameters:
        args (Namespace): Namespace object containing arguments.
            grayscale (bool): Whether to convert the images to grayscale.
        aug_list (List): The list of transformation functions for data augmentation.

    Returns:
        List: The updated list of transformation functions with normalization applied.

    Examples:
        >>> from argparse import Namespace
        >>> args = Namespace(grayscale=True)
        >>> aug_list = [v2.RandomResizedCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        >>> updated_aug_list = apply_normalization(args, aug_list)

        If 'grayscale' is True, the updated_aug_list will contain additional transformations:
        [v2.RandomResizedCrop(224), v2.Grayscale(), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
         v2.Normalize(mean=[0.5], std=[0.5])]

        If 'grayscale' is False, the updated_aug_list will contain different normalization values:
        [v2.RandomResizedCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    """
    if args.grayscale:
        aug_list.append(v2.Grayscale())

    normalization_mean = [0.5] if args.grayscale else IMAGENET_DEFAULT_MEAN
    normalization_std = [0.5] if args.grayscale else IMAGENET_DEFAULT_STD

    aug_list.extend([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalization_mean, std=normalization_std),
    ])

    return aug_list


def get_augmentation_by_type(args: Namespace) -> Callable:
    """
    Returns an augmentation transform based on the specified augmentation type.

    Parameters:
        args (Namespace): A namespace object containing the following attributes:
            aug_type (str): The type of augmentation to apply. Must be one of "trivial", "augmix", or "rand".
            mag_bins (int): The number of magnitude bins for augmentation (used for "trivial" and "rand" types).
            interpolation (str): The interpolation method for resizing and cropping (e.g., "bilinear").

    Returns:
        Callable: A callable augmentation transform based on the specified augmentation type.

    Raises:
        ValueError: If the specified augmentation type is invalid.

    Examples:
        >>> from argparse import Namespace
        >>> args = Namespace(aug_type="trivial", mag_bins=30, interpolation="bilinear")
        >>> augmentation = get_augmentation_by_type(args)
        >>> isinstance(augmentation, v2.TrivialAugmentWide)
        True
    """
    valid_aug_types = ["trivial", "augmix", "rand", "auto"]
    if args.aug_type not in valid_aug_types:
        raise ValueError(f"Invalid augmentation type: '{args.aug_type}'. "
                         f"Valid options are: {', '.join(valid_aug_types)}")

    interpolation_mode = f.InterpolationMode(args.interpolation)

    if args.aug_type == "trivial":
        # Trivial augmentation
        return v2.TrivialAugmentWide(num_magnitude_bins=args.mag_bins, interpolation=interpolation_mode)
    elif args.aug_type == "augmix":
        # AugMix augmentation
        return v2.AugMix(interpolation=interpolation_mode)
    elif args.aug_type == "rand":
        # RandAugment augmentation
        return v2.RandAugment(num_magnitude_bins=args.mag_bins, interpolation=interpolation_mode)
    elif args.aug_type == "auto":
        # AutoAugment augmentation
        return v2.AutoAugment(interpolation=interpolation_mode)


def get_data_augmentation(args: Namespace) -> Dict[str, Callable]:
    """
    Returns data augmentation transforms for training and validation sets.

    Parameters:
        args (Namespace): A namespace object containing the following attributes:
            crop_size (int): The size of the crop for the training and validation sets.
            val_resize (int): The target size for resizing the validation images.
                              The validation images will be resized to this size while maintaining their aspect ratio.
            interpolation (int): The interpolation method for resizing and cropping.
            hflip (float): The probability of applying random horizontal flip to the training set, a float between 0 and 1.
            aug_type (str): The type of augmentation to apply to the training set.
                             Must be one of "trivial", "augmix", or "rand".

    Returns:
        Dict[str, Callable]: A dictionary of data augmentation transforms for the training and validation sets.

    Raises:
        ValueError: If the provided augmentation type is not one of the supported types.

    Examples:
        >>> args = Namespace( \
                crop_size=224, \
                val_resize=256,\
                interpolation="bilinear", \
                hflip=0.5, \
                aug_type="augmix", \
                mag_bins=30, \
                grayscale=False, \
            )

        >>> data_transforms = get_data_augmentation(args)
        >>> data_transforms["train"]
        Compose(
            RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=warn),
            RandomHorizontalFlip(p=0.5)
            AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=True, interpolation=InterpolationMode.BILINEAR, fill=None),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    """
    supported_aug_types = ["trivial", "augmix", "rand", "auto"]

    if args.aug_type not in supported_aug_types:
        raise ValueError(f"Unsupported augmentation type: '{args.aug_type}'. "
                         f"Supported types are: {', '.join(supported_aug_types)}")

    train_aug = [
        v2.RandomResizedCrop(
            args.crop_size, interpolation=f.InterpolationMode(args.interpolation), antialias=True
        ),
        v2.RandomHorizontalFlip(p=args.hflip), get_augmentation_by_type(args)]

    norm_train_aug = apply_normalization(args, train_aug)
    if args.rand_erase_prob:
        norm_train_aug.extend([
            v2.RandomErasing(p=args.rand_erase_prob),
            v2.ToPureTensor()
        ])
    else:
        norm_train_aug.append(v2.ToPureTensor())
    train_transform = v2.Compose(norm_train_aug)

    val_aug = [
        v2.Resize(args.val_resize, interpolation=f.InterpolationMode(args.interpolation), antialias=True),
        v2.CenterCrop(args.crop_size),
    ]
    norm_val_aug = apply_normalization(args, val_aug)
    norm_val_aug.append(v2.ToPureTensor())
    val_transform = v2.Compose(norm_val_aug)

    return {"train": train_transform, "val": val_transform}


def to_channels_first(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor from channels-last format to channels-first format.

    Parameters:
        image (torch.Tensor): A 4-D or 3-D image tensor.

    Returns:
        torch.Tensor: The image tensor in channels-first format.

    Examples:
        Converting a 4-D image tensor (batch_size, height, width, channels) to channels-first format:

        >>> import torch
        >>> image = torch.randn(32, 64, 64, 3)  # Example 4-D image tensor with channels last
        >>> converted_image = to_channels_first(image)
        >>> converted_image.shape
        torch.Size([32, 3, 64, 64])

        Converting a 3-D image tensor (height, width, channels) to channels-first format:

        >>> image = torch.randn(64, 64, 3)  # Example 3-D image tensor with channels last
        >>> converted_image = to_channels_first(image)
        >>> converted_image.shape
        torch.Size([3, 64, 64])
    """
    return image.permute(0, 3, 1, 2) if image.dim() == 4 else image.permute(2, 0, 1)


def to_channels_last(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor from the channels-first format to the channels-last format.

    Parameters:
        image (torch.Tensor): A 4-D or 3-D image tensor in channels-first format.

    Returns:
        torch.Tensor: The image tensor in channels-last format.

    Examples:
        Converting a 4-D image tensor (batch_size, channels, height, width) to channels-last format:

        >>> import torch
        >>> image = torch.randn(1, 3, 32, 32)  # Channels-first format
        >>> image_channels_last = to_channels_last(image)
        >>> image_channels_last.shape
        torch.Size([1, 32, 32, 3])

        Converting a 3-D image tensor (channels, height, width) to channels-last format:

        >>> image = torch.randn(3, 32, 32)  # Channels-first format
        >>> image_channels_last = to_channels_last(image)
        >>> image_channels_last.shape
        torch.Size([32, 32, 3])
    """
    return image.permute(0, 2, 3, 1) if image.dim() == 4 else image.permute(1, 2, 0)


def convert_to_onnx(
        args: Namespace,
        model_name: str,
        checkpoint_path: str,
        num_classes: int,
) -> None:
    """
    Convert a PyTorch model to ONNX format.

    Parameters:
        args: A namespace object containing the following attributes:
            crop_size (int, optional): The size of the crop for the inference dataset/image. Default: None.
        model_name (str): The name of the model.
        checkpoint_path (str): The path to the PyTorch checkpoint.
        num_classes (int): The number of classes in the dataset.

    Examples:
        # >>> args = Namespace(crop_size=224, dropout=0.2, grayscale=False, feat_extract=True)
        # >>> model_name = "xcit_nano_12_p8_224"
        # >>> checkpoint_path = "best_model.pth"
        # >>> num_classes = 2
        # >>> convert_to_onnx(args, model_name, checkpoint_path, num_classes)
    """
    torch.jit.enable_onednn_fusion(True)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

    model = timm.create_model(
        model_name,
        scriptable=True,
        drop_rate=args.dropout,
        drop_path_rate=args.dropout,
        in_chans=1 if args.grayscale else 3,
        num_classes=num_classes
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "n_averaged" in checkpoint["model"]:
        del checkpoint["model"]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["model"], "module.")

    model.load_state_dict(checkpoint["model"])

    batch_size = 1
    input_channels = 1 if args.grayscale else 3
    dummy_input = torch.randn(batch_size, input_channels, args.crop_size, args.crop_size, requires_grad=True)

    filename = os.path.join(os.path.dirname(checkpoint_path), "best_model.onnx")

    with torch.no_grad():
        model.eval()
        model = torch.jit.trace(model, dummy_input)
        model = torch.jit.freeze(model)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"ONNX model saved to {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to export the model to ONNX: {e}")


def get_explanation_transforms() -> Tuple[v2.Compose, v2.Compose]:
    """
    Get transforms for preprocessing and inverse preprocessing of images used in an explanation pipeline.

    Returns:
        Tuple[v2.Compose, v2.Compose]: A tuple of two torchvision.transforms.v2.Compose objects.
            - The first transform is used for preprocessing an image for explanation.
            - The second transform is used for inverse preprocessing to revert the explanation to the original image.

    Examples:
        transform, inv_transform = get_explanation_transforms()
        preprocessed_image = transform(original_image)
        inverted_image = inv_transform(explanation_image)
    """
    transform = v2.Compose([
        v2.Lambda(to_channels_first),
        v2.Lambda(lambda image: image * (1 / 255)),
        v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        v2.Lambda(to_channels_last),
    ])

    inv_transform = v2.Compose([
        v2.Lambda(to_channels_first),
        v2.Normalize(
            mean=(-1 * np.array(IMAGENET_DEFAULT_MEAN) / np.array(IMAGENET_DEFAULT_STD)).tolist(),
            std=(1 / np.array(IMAGENET_DEFAULT_STD)).tolist()
        ),
        v2.Lambda(to_channels_last),
    ])

    return transform, inv_transform


def collate_fn(examples) -> Dict:
    """
    Collates a list of examples into batches by stacking pixel values of images and creating a tensor for labels.

    Parameters:
        examples (list): A list of examples, each containing a dictionary with "pixel_values" and "labels" keys.

    Returns:
        Dict: A dictionary containing the batched pixel values and labels.

    Examples:
        >>> examples = [ \
                {"pixel_values": torch.tensor([1, 2, 3]), "labels": 0}, \
                {"pixel_values": torch.tensor([4, 5, 6]), "labels": 1}, \
                {"pixel_values": torch.tensor([7, 8, 9]), "labels": 2} \
                ]
        >>> batch = collate_fn(examples)
        >>> print(batch)
        {'pixel_values': tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]), 'labels': tensor([0, 1, 2])}
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def get_classes(dataset: torch.utils.data.Dataset) -> List[str]:
    """
    Get a list of the classes in a dataset.

    Parameters:
        dataset: dataset to get classes from.

    Returns:
        List[str]: A sorted list of the classes in the dataset.

    Examples:
        >>> from argparse import Namespace

        >>> args = Namespace( \
                dataset = "cifar10", \
                dataset_kwargs="", \
                crop_size=224, \
                val_resize=256,\
                interpolation="bilinear", \
                hflip=0.5, \
                aug_type="augmix", \
                mag_bins=30, \
                grayscale=False, \
                )

        >>> data_transforms = get_data_augmentation(args)
        >>> dataset = load_image_dataset(args)

        >>> train_dataset, val_dataset, test_dataset = preprocess_train_eval_data(dataset, data_transforms)

        >>> classes = get_classes(train_dataset)
        >>> print(classes)
        ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """
    classes = dataset.features["labels"].names

    return sorted(classes)


def get_matching_model_names(args: Namespace) -> List[str]:
    """
    Get a list of model names matching the given image crop size and model size or submodule name.

    Parameters:
        args:
            crop_size (int): Image size the models should be trained on.
            model_size (str): Size of the model (e.g., "tiny", "small", etc.).
            module (str): A submodule for selecting models

    Returns:
        List[str]: A list of model names that can be used for training.

    Examples:
        >>> from argparse import Namespace

        ### Loading models based on model size
        >>> args = Namespace(model_size="nano", crop_size="224", module=None)
        >>> get_matching_model_names(args)
        ['coatnet_nano_rw_224.sw_in1k', 'coatnet_rmlp_nano_rw_224.sw_in1k', 'coatnext_nano_rw_224.sw_in1k', 'xcit_nano_12_p8_224.fb_dist_in1k', 'xcit_nano_12_p8_224.fb_in1k', 'xcit_nano_12_p16_224.fb_dist_in1k', 'xcit_nano_12_p16_224.fb_in1k']

        # Loading models based on specific submodule
        >>> args = Namespace(crop_size=224, module="edgenext")
        >>> get_matching_model_names(args)
        ['edgenext_base.in21k_ft_in1k', 'edgenext_base.usi_in1k', 'edgenext_small.usi_in1k', 'edgenext_small_rw.sw_in1k', 'edgenext_x_small.in1k', 'edgenext_xx_small.in1k']
    """

    def filter_models(models: List[str], crop_size: int) -> List[str]:
        """
        Filter a list of model names based on crop size.

        This function iterates through the list of model names and removes
        those models that do not contain the given crop size in their name
        and have numeric values greater than the crop size in their suffixes.

        Parameters:
            models (List[str]): A list of model names.
            crop_size (int): The crop size to be checked against the models.

        Returns:
            List[str]: A filtered list of model names after the recurring filtering process.

        Examples:
            >>> models = ['flexivit_base.300ep_in1k', 'vit_small_patch16_224.augreg_in1k', 'vit_tiny_patch16_384.augreg_in21k_ft_in1k']
            >>> crop_size = 224
            >>> filtered_models = filter_models(models, crop_size)
            >>> print(filtered_models)
            ['flexivit_base.300ep_in1k', 'vit_small_patch16_224.augreg_in1k']
        """
        changed = True
        while changed:
            changed = False
            for model in models.copy():
                lhs = model.split(".")[0].split("_")[-1]
                rhs = model.split(".")[-1].split("_")[-1]
                if str(crop_size) not in model:
                    if model.isalpha():
                        continue
                    if lhs.isnumeric():
                        if int(lhs) > crop_size:
                            models.remove(model)
                            changed = True
                    if rhs.isnumeric():
                        if int(rhs) > crop_size:
                            models.remove(model)
                            changed = True
        return models

    def is_matching_model(name: str) -> bool:
        return str(args.crop_size) in name and args.model_size in name

    if args.module:
        model_names = timm.list_models(pretrained=True, module=args.module)
        matching_models = filter_models(model_names, args.crop_size)
        matching_models = remove_items(matching_models, args.filter_module)
    else:
        model_names = timm.list_models(pretrained=True)

        models_to_remove = {
            "tiny": {"deit_tiny_distilled_patch16_224.fb_in1k", "swin_s3_tiny_224.ms_in1k"},
            "small": {"deit_small_distilled_patch16_224.fb_in1k"},
            "base": {"deit_base_distilled_patch16_224.fb_in1k", "vit_base_patch8_224.augreg2_in21k_ft_in1k"}
        }
        matching_models = [name for name in model_names if is_matching_model(name)]
        matching_models = [name for name in matching_models if name not in models_to_remove.get(args.model_size, set())]
        matching_models = remove_items(matching_models, args.filter_models)
        
    return matching_models


def prune_model(model: nn.Module, pruning_rate: float) -> List[Tuple[nn.Module, str]]:
    """
    Applies global unstructured pruning to the model.

    Parameters:
        model (nn.Module): The model to be pruned.
        pruning_rate (float): The fraction of weights to be pruned.

    Returns:
        List[Tuple[nn.Module, str]]: A list of tuples containing the pruned modules and parameter names.

    Examples:
        >>> import torch
        >>> import torch.nn as nn

        >>> class SimpleCNN(nn.Module):
        ...     def __init__(self, num_classes):
        ...         super(SimpleCNN, self).__init__()
        ...         # First convolutional layer
        ...         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        ...         self.relu1 = nn.ReLU()
        ...         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        ...
        ...         # Second convolutional layer
        ...         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        ...         self.relu2 = nn.ReLU()
        ...         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        ...
        ...         # Fully connected layers
        ...         self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Assuming input images are 32x32 pixels
        ...         self.relu3 = nn.ReLU()
        ...         self.fc2 = nn.Linear(128, num_classes)
        ...
        ...     def forward(self, x):
        ...         x = self.pool1(self.relu1(self.conv1(x)))
        ...         x = self.pool2(self.relu2(self.conv2(x)))
        ...         x = x.view(x.size(0), -1)  # Flatten the tensor
        ...         x = self.relu3(self.fc1(x))
        ...         x = self.fc2(x)
        ...         return x

        >>> # Instantiate the model with the number of output classes
        >>> num_classes = 10  # Example: CIFAR-10 has 10 classes
        >>> model = SimpleCNN(num_classes)

        >>> pruning_rate = 0.5
        >>> pruned_params = prune_model(model, pruning_rate)
        >>> print(pruned_params)
    """

    parameters_to_prune = [
        (module, 'weight') for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    # noinspection PyTypeChecker
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate
    )
    return parameters_to_prune


def remove_pruning_reparam(parameters_to_prune: List[Tuple[nn.Module, str]]) -> None:
    """
    Removes pruning re-parametrization for each module and parameter in the provided list.

    Parameters:
        parameters_to_prune (List[Tuple[nn.Module, str]]): List of module and parameter names to remove pruning re-parametrization.

    Examples:
        >>> import timm
        >>> model = timm.create_model("fastvit_t12.apple_in1k")
        >>> parameters_to_prune = prune_model(model, pruning_rate=0.2)
        >>> remove_pruning_reparam(parameters_to_prune)
    """
    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)


def get_pretrained_model(args: Namespace, model_name: str, num_classes: int) -> Any:
    """
    Returns a pretrained model with a new head and the specified model name.

    The head of the model is replaced with a new linear layer with the given
    number of classes.

    Parameters:
        args (Namespace): A namespace object containing the following attributes:
            - feat_extract (bool): Whether to freeze the parameters of the model.
            - dropout (float): The dropout rate.
        model_name (str): The name of the model to be created using the `timm` library.
        num_classes (int): The number of classes for the new head of the model.

    Returns:
        nn.Module: The modified model with the new head.

    Examples:
        >>> args = Namespace(feat_extract=True, dropout=0.5, grayscale=False)
        >>> model = get_pretrained_model(args, "tf_efficientnet_b0.ns_jft_in1k", num_classes=10)
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        drop_rate=args.dropout,
        drop_path_rate=args.dropout,
        in_chans=1 if args.grayscale else 3,
        num_classes=num_classes
    )

    if args.feat_extract:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model


def calculate_class_weights(dataset: datasets.arrow_dataset.Dataset) -> torch.Tensor:
    """
    Returns the class weights for the given image classification dataset.

    The class weights are calculated as the inverse frequency of each class in the train dataset.

    Parameters:
        dataset (datasets.arrow_dataset.Dataset): A HuggingFace image classification dataset.

    Returns:
        torch.Tensor: A tensor of class weights.

    Examples:
        >>> from argparse import Namespace
        >>> import numpy as np

        >>> args = Namespace(dataset = "cifar10", dataset_kwargs = "")

        >>> image_dataset = load_image_dataset(args)
        >>> class_weights = calculate_class_weights(image_dataset)
        >>> print(class_weights)
        tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    """
    # noinspection PyTypeChecker
    labels = dataset['train']['labels']
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = torch.tensor(total_samples / (class_counts * len(class_counts)), dtype=torch.float32)

    return class_weights


# adapted from https://github.com/pytorch/vision/blob/a5035df501747c8fc2cd7f6c1a41c44ce6934db3/references
# /classification/utils.py#L272
def average_checkpoints(checkpoint_paths: List[str]) -> OrderedDict:
    """
    Averages the parameters of multiple checkpoints.

    Parameters:
        checkpoint_paths (List[str]): List of file paths to the input checkpoint files.

    Returns:
        OrderedDict: Averaged parameters in the form of an ordered dictionary.

    Raises:
        KeyError: If the checkpoints have different sets of parameters.

    Examples:
        >>> from glob import glob

        >>> ckpts = glob("something/xcit_nano_12_p8_224/best_model_*.pth")
        >>> averaged_ckpt = average_checkpoints(ckpts)
        >>> averaged_ckpt.keys()
        odict_keys(['model'])

    """
    averaged_params = OrderedDict()
    num_checkpoints = len(checkpoint_paths)
    params_keys = None

    for checkpoint_path in checkpoint_paths:
        with open(checkpoint_path, "rb") as f_in:
            state = torch.load(
                f_in,
                map_location=lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            )

        if params_keys is None:
            params_keys = list(state["model"].keys())
        elif params_keys != list(state["model"].keys()):
            raise KeyError(
                f"For checkpoint {checkpoint_path}, expected list of params: {params_keys}, "
                f"but found: {list(state['model'].keys())}"
            )

        for k, p in state["model"].items():
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in averaged_params:
                averaged_params[k] = p.clone()
            else:
                averaged_params[k] += p

    for k, v in averaged_params.items():
        if v.is_floating_point():
            averaged_params[k].div_(num_checkpoints)
        else:
            averaged_params[k] //= num_checkpoints

    new_state = {"model": averaged_params}
    return OrderedDict(new_state)


def get_optimizer(args: Namespace, model: torch.nn.Module) -> optim.Optimizer:
    """
    Returns an optimizer object based on the provided optimization algorithm name.

    Parameters:
        args (Namespace): A namespace object containing the following attributes:
            - opt_name (str): The name of the optimization algorithm.
            - lr (float): The learning rate for the optimizer.
            - wd (float): The weight decay for the optimizer.
        model (torch.mo ): A list of parameters for the optimizer.

    Returns:
        optim.Optimizer: An optimizer object of the specified type.

    Examples:
        >>> from argparse import Namespace
        >>> import torch
        >>> model = torch.nn.Linear(10, 10)
        >>> args = Namespace(opt_name="sgd", lr=0.01, wd=0.0001)
        >>> optimizer = get_optimizer(args, model)
    """
    optimizer = create_optimizer_v2(model_or_params=model, opt=args.opt_name, lr=args.lr, weight_decay=args.wd)
    return optimizer


def get_lr_scheduler(args: Namespace,
                     optimizer: optim.Optimizer,
                     num_iters: int
                     ) -> Union[lr_scheduler.SequentialLR, lr_scheduler.LRScheduler, None]:
    """
    Returns a learning rate scheduler object based on the provided scheduling algorithm name.

    Parameters:
        args (Namespace): A namespace object containing the following attributes:
            - sched_name (str): The name of the scheduling algorithm.
            - warmup_decay (float): The decay rate for the warmup scheduler.
            - warmup_epochs (int): The number of epochs for the warmup scheduler.
            - step_size (int): The step size for the StepLR scheduler.
            - gamma (float): The gamma for the StepLR scheduler.
            - epochs (int): The total number of epochs for training.
            - eta_min (float): The minimum learning rate for the CosineAnnealingLR scheduler.
        optimizer (optim.Optimizer): The optimizer object to be used with the scheduler.
        num_iters (int): The total number of iterations in an epoch.

    Returns:
        Union[lr_scheduler.SequentialLR, lr_scheduler.LRScheduler, None]: A learning rate scheduler object of the
        specified type, or None if the sched_name is not recognized.

     Raises:
        ValueError: If an unsupported scheduler type is provided in args.sched_name.

    Examples:
        # Obtain a learning rate scheduler based on the provided args and optimizer:
        >>> from argparse import Namespace
        >>> import torch
        >>> model = torch.nn.Linear(10, 10)
        >>> args = Namespace(opt_name="sgd", lr = 0.01, wd = 0.0001, sched_name = "cosine", warmup_decay = 0.1, warmup_epochs = 5, step_size = 10, gamma = 0.5, epochs = 50, eta_min = 0.001)
        >>> optimizer = get_optimizer(args, model)
        >>> scheduler = get_lr_scheduler(args, optimizer, num_iters = 100)
        >>> print(scheduler.state_dict()["_last_lr"])
        [0.001, 0.001]
    """
    scheduler_options = {
        "step": lambda: lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma),
        "cosine": lambda: lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                                                         eta_min=args.eta_min),
        "cosine_wr": lambda: lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, eta_min=args.eta_min),
        "one_cycle": lambda: lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=num_iters * args.epochs)
    }

    if args.sched_name in scheduler_options:
        scheduler = scheduler_options[args.sched_name]()
    else:
        raise ValueError(f"Unsupported scheduler type: {args.sched_name}")

    if args.warmup_epochs > 0:
        warmup_lr = lr_scheduler.LinearLR(optimizer, start_factor=args.warmup_decay, total_iters=args.warmup_epochs)
        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr, scheduler],
                                              milestones=[args.warmup_epochs])

    return scheduler


def gather_for_metrics(accelerator: Accelerator, output: Tensor, labels: Tensor):
    """
    Helper function to efficiently gather predictions and labels across multiple accelerator devices for computing metrics.

    Args:
        accelerator (accelerate.Accelerator): The PyTorch accelerator object.
        output (torch.Tensor): The model's output predictions.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the aggregated predictions and labels.
    """

    return accelerator.gather_for_metrics(output), accelerator.gather_for_metrics(labels)


class CreateImgSubclasses:
    def __init__(self, img_src: str, img_dest: str) -> None:
        """
        Initialize the object with the path to the source and destination directories.

        Parameters:
            img_src (str): Path to the source directory.
            img_dest (str): Path to the destination directory.
        """
        self.img_src = img_src
        self.img_dest = img_dest

    def get_image_classes(self) -> List[str]:
        """
        Get a list of the classes in the source directory.

        Returns:
            List[str]: A list of the classes in the source directory.
        """
        file_set = set()
        pattern = r'[^A-Za-z]+'
        for file in glob(os.path.join(self.img_src, "*")):
            file_name = os.path.basename(file).split(".")[0]
            cls_name = re.sub(pattern, '', file_name)
            file_set.add(cls_name)

        return list(file_set)

    def create_class_dirs(self, class_names: List[str]) -> None:
        """
        Create directories for each class in `class_names` under `self.img_dest` directory.

        Parameters:
            class_names (List[str]): A list of strings containing the names of the image classes.
        """
        for dir_name in class_names:
            os.makedirs(os.path.join(self.img_dest, dir_name), exist_ok=True)

    def copy_images_to_dirs(self) -> None:
        """
        Copy images from `self.img_src` to corresponding class directories in `self.img_dest`.

        The image file is copied to the class directory whose name is contained in the file name.
        If no class name is found in the file name, the file is not copied.
        """
        class_names = self.get_image_classes()
        for file in glob(os.path.join(self.img_src, "*")):
            for dir_name in class_names:
                if dir_name in file:
                    try:
                        shutil.copy(file, os.path.join(self.img_dest, dir_name))
                    except Exception as e:
                        print(f"Error copying {file} to {dir_name}: {str(e)}")
                    break


def create_train_val_test_splits(img_src: str, img_dest: str, ratio: tuple) -> None:
    """
    Split images from `img_src` directory into train, validation, and test sets and save them in `img_dest`
    directory. This will save the images in the appropriate directories based on the train-val-test split ratio.

    Parameters:
        img_src (str): The source directory containing the images to be split.
        img_dest (str): The destination directory where the split images will be saved.
        ratio (tuple): The train, val, test splits. E.g (0.8, 0.1, 0.1)

    Examples:
        # Split images from "data/images" into train, validation, and test sets with a split ratio of (0.8, 0.1, 0.1)
        # and save them in the "data/splits" directory:

        # create_train_val_test_splits("data/images", "data/splits", ratio=(0.8, 0.1, 0.1))
    """
    try:
        # Use multiprocessing to parallelize the splitting process
        multiprocessing.freeze_support()
        splitfolders.ratio(img_src, output=img_dest, seed=333777999, ratio=ratio)
    except Exception as e:
        print(f"Error during image splitting: {str(e)}")
