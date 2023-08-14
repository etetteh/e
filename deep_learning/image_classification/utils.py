import os
import re
import json
import random
import shutil

from glob import glob
from os import PathLike
from pathlib import Path
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import timm
import torch
import datasets
import numpy as np
import splitfolders
import torch.nn.utils.prune as prune
import torch.optim.lr_scheduler as lr_scheduler

from datasets import load_dataset
from torch.optim import swa_utils
from torch import nn, optim, Tensor
from torch.distributions import Beta
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as f
from timm.optim import create_optimizer_v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def print_header(*args: Union[str, int, float]) -> None:
    """
    Print one or more arguments as a single line of text to the standard output.

    Parameters:
        *args: The arguments to be printed.

    Example:
        >>> print_header("Hello", "World", 2023)
        Output: Hello World 2023

        >>> print_header("The answer is", 42, "and the value of pi is", 3.14159)
        Output: The answer is 42 and the value of pi is 3.14159

        >>> print_header("A single value:", 10)
        Output: A single value: 10
    """
    message = " ".join(map(str, args))
    print(message)


def print_heading(message: str) -> None:
    """
    Print a formatted heading with the given message.

    Parameters:
        message (str): The message to be included in the heading.

    Example:
        >>> print_heading("Welcome to the Chatbot")
        Output:
        =========================
        Welcome to the Chatbot
        =========================

        >>> print_heading("Instructions")
        Output:
        =============
        Instructions
        =============

        >>> print_heading("Important Notice")
        Output:
        =================
        Important Notice
        =================
    """
    line = "=" * len(message) + "=="
    print_header(line)
    print_header(message)
    print_header(line)


# noinspection PyShadowingNames
def get_model_run_id(run_ids: Dict[str, str], model_name: str) -> Optional[str]:
    """
    Get the run ID of a specific model from a dictionary of run IDs.

    Parameters:
        run_ids (Dict[str, str]): A dictionary mapping model names to run IDs.
        model_name (str): The name of the model for which to retrieve the run ID.

    Returns:
        Optional[str]: The run ID of the specified model, or None if the model is not in the dictionary.

    Example:
        >>> run_ids = {"model1": "1234", "model2": "5678", "model3": "9012"}  # Example dictionary of run IDs

        >>> model_name = "model2"  # Example model name to retrieve the run ID
        >>> run_id = get_model_run_id(run_ids, model_name)
        >>> print(run_id)
        Output: 5678

        >>> model_name = "model4"  # Non-existing model name
        >>> run_id = get_model_run_id(run_ids, model_name)
        >>> print(run_id)
        Output: None
    """
    try:
        run_id = run_ids[model_name]
    except KeyError:
        return None
    return run_id


def write_dictionary_to_json(dictionary: Dict, file_path: str) -> None:
    """
    Write a dictionary object to a JSON file at the given file path.
    If the file already exists, its content will be overwritten.

    Parameters:
        dictionary (Dict): The dictionary object to be written to the JSON file.
        file_path (str): The path to the JSON file.

    Returns:
        None

    Example:
        >>> dictionary = {"key": "value"}  # Example dictionary object to be written
        >>> file_path = "data.json"  # Example file path
        >>> write_dictionary_to_json(dictionary, file_path)

        # The data.json file will contain the following content:
        # {
        #     "key": "value"
        # }
    """
    with open(file_path, "w") as file_out:
        json.dump(dictionary, file_out, indent=4)


def append_dictionary_to_json_file(new_dict: Dict, file_path: str) -> None:
    # noinspection PyShadowingNames
    """
    Append a dictionary to a JSON file at the given file path.
    If the file does not exist, it will be created.

    Parameters:
        new_dict (Dict): The dictionary to be appended to the JSON file.
        file_path (str): The path to the JSON file.

    Returns:
        None

    Example:
        >>> new_dict = {"key": "value"}  # Example dictionary to be appended
        >>> file_path = "data.json"  # Example file path
        >>> append_dictionary_to_json_file(new_dict, file_path)

        # If data.json file does not exist previously, it will be created and contain the following content:
        # {
        #     "key": "value"
        # }

        # If data.json file already exists with content {"existing_key": "existing_value"},
        # after appending new_dict, the content will be updated to:
        # {
        #     "existing_key": "existing_value",
        #     "key": "value"
        # }
    """
    if os.path.isfile(file_path):
        data = read_json_file(file_path)
    else:
        data = {}

    data.update(new_dict)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a JSON file from the given file path and return the data as a dictionary.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The data contained in the JSON file as a dictionary.

    Example:
        >>> file_path = "data.json"  # Example file path
        >>> data = read_json_file(file_path)
        >>> print(data)
        Output: {'key': 'value'}

        # Assuming data.json contains the following JSON content:
        # {
        #     "key": "value"
        # }
    """
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def read_json_lines_file(file_path: str) -> List[Union[dict, Any]]:
    """
    Read and parse a JSON Lines file from the given file path and return the data as a list.

    Parameters:
        file_path (str): The path to the JSON Lines file.

    Returns:
        List[Union[dict, Any]]: The data contained in the JSON Lines file as a list.

    Example:
        >>> file_path = "data.jsonl"  # Example file path
        >>> data = read_json_lines_file(file_path)
        >>> print(data)
        Output: [{'key': 'value'}, {'key2': 42}, 'text line', 3.14]

        # Assuming data.jsonl contains the following JSON Lines content:
        # {"key": "value"}
        # {"key2": 42}
        # "text line"
        # 3.14
    """
    with open(file_path, "r") as file_in:
        data = [json.loads(line) for line in file_in]
        return data


def keep_recent_files(directory: str, num_files_to_keep: int) -> None:
    """
    Sorts files in the specified directory based on modification time and keeps
    the most recent files. Older files beyond the specified number of files to keep
    will be removed.

    Parameters:
        directory (str): The path to the directory containing the files.
        num_files_to_keep (int): The number of most recent files to keep.

    Returns:
        None

    Example:
        >>> directory_path = "/path/to/directory"
        >>> num_files_to_keep = 10
        >>> keep_recent_files(directory_path, num_files_to_keep)
    """
    file_list = glob(os.path.join(directory, "best_model_", "*"))
    sorted_files = sorted(file_list, key=os.path.getmtime, reverse=True)
    files_to_remove = sorted_files[num_files_to_keep:]

    for file_to_remove in files_to_remove:
        try:
            os.remove(file_to_remove)
            print(f"Removed file: {file_to_remove}")
        except OSError as e:
            print(f"Error while removing {file_to_remove}: {e}")


def set_seed_for_worker(worker_id: Optional[int]) -> Optional[int]:
    """
    Sets the seed for NumPy and Python's random module for the given worker.
    If no worker ID is provided, uses the initial seed for PyTorch and returns None.

    Parameters:
        worker_id (Optional[int]): The ID of the worker.

    Returns:
        Optional[int]: The seed used for the worker, or None if no worker ID was provided.

    Example:
        >>> worker_id = 1  # Example worker ID
        >>> seed = set_seed_for_worker(worker_id)
        >>> print(seed)  # Seed used for the worker
        Output: 1

        >>> seed = set_seed_for_worker(None)  # No worker ID provided
        >>> print(seed)  # Initial seed for PyTorch
        Output: <initial_seed_value>
    """
    if worker_id is not None:
        worker_seed = worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        return worker_seed
    else:
        initial_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(initial_seed)
        random.seed(initial_seed)
        return None


# noinspection PyTypeChecker
def load_image_dataset(dataset: Union[str, os.PathLike]) -> datasets.arrow_dataset.Dataset:
    """
    Load an image dataset using Hugging Face's 'datasets' library.

    Parameters:
        dataset (str or os.PathLike): The path to a local dataset directory or a Hugging Face dataset name
                                    (or an HTTPS URL for a remote dataset).

    Returns:
        datasets.arrow_dataset.Dataset: The loaded image dataset.

    Raises:
        ValueError: If the provided dataset is not a valid path to a directory, a string (Hugging Face dataset name),
                    or an HTTPS URL for a remote dataset.

     Example:
         >>> dataset_name = "mnist"
         >>> loaded_dataset = load_image_dataset(dataset_name)
         >>> print(loaded_dataset)
         Dataset(features: {'image': Image(shape=(28, 28, 1), dtype=torch.uint8), 'label': ClassLabel(
                            shape=(), dtype=int64)}, num_rows: 70000)

        >>> local_path = "/path/to/local/image_dataset"
        >>> loaded_dataset = load_image_dataset(local_path)
        >>> print(loaded_dataset)
        Dataset(features: {'image': Image(shape=(None, None, 3), dtype=uint8),
                           'label': ClassLabel(shape=(), dtype=int64)}, num_rows: 1000)

        >>> remote_url = "https://url.to.remote/image_dataset"
        >>> loaded_dataset = load_image_dataset(remote_url)
        >>> print(loaded_dataset)
        Dataset(features: {'image': Image(shape=(None, None, 3), dtype=uint8)}, num_rows: 5000)
    """
    if isinstance(dataset, Path) and os.path.isdir(dataset):
        image_dataset = load_dataset("imagefolder", data_dir=dataset)
    # elif isinstance(dataset, str) and (dataset.startswith("https") or dataset.endswith(".zip")):
    #     image_dataset = load_dataset("imagefolder", data_files=dataset)
    elif isinstance(dataset, str):
        image_dataset = load_dataset(dataset, name=None)
    else:
        raise ValueError("Dataset should be a path to a local dataset on disk, a dataset name of an image dataset "
                         "from Hugging Face datasets, or an HTTPS URL for a remote dataset.")

    image_column_names = image_dataset.column_names["train"]
    if "img" in image_column_names:
        image_dataset = image_dataset.rename_columns({"img": "image"})
    if "label" in image_column_names:
        image_dataset = image_dataset.rename_columns({"label": "labels"})
    if "fine_label" in image_column_names:
        image_dataset = image_dataset.rename_columns({"fine_label": "labels"})

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


def apply_fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """
    Generate an adversarial example using the Fast Gradient Sign Method (FGSM).

    Parameters:
        image (torch.Tensor): The input image.
        epsilon (float): Perturbation magnitude for generating adversarial examples.
        data_grad (torch.Tensor): Gradient of the loss with respect to the image.

    Returns:
        torch.Tensor: Adversarial example perturbed using FGSM.

    Example:
        >>> image = torch.rand(1, 3, 32, 32)  # A random image of size (1, 3, 32, 32)
        >>> epsilon = 0.05  # Perturbation magnitude
        >>> data_grad = torch.randn(1, 3, 32, 32)  # Gradient of loss w.r.t. image
        >>> adversarial_image = fgsm_attack(image, epsilon, data_grad)
    """
    sign_data_grad = data_grad.sign()
    adversarial_image = image + epsilon * sign_data_grad
    adversarial_image = torch.clamp(adversarial_image, min=0, max=1)
    return adversarial_image


def apply_normalization(args: Namespace, aug_list: List) -> List:
    """
    Apply normalization to the augmentation list based on grayscale conversion.

    Parameters:
        args (Namespace): Namespace object containing arguments.
            grayscale (bool): Whether to convert the images to grayscale.
        aug_list (List): The list of transformation functions for data augmentation.

    Returns:
        List: The updated list of transformation functions with normalization applied.

    Example:
        >>> from argparse import Namespace
        >>> args = Namespace(grayscale=True)
        >>> aug_list = [transforms.RandomResizedCrop(224), transforms.ToTensor()]
        >>> updated_aug_list = apply_normalization(args, aug_list)

        If 'grayscale' is True, the updated_aug_list will contain additional transformations:
        [transforms.RandomResizedCrop(224), transforms.Grayscale(), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])]

        If 'grayscale' is False, the updated_aug_list will contain different normalization values:
        [transforms.RandomResizedCrop(224), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    """
    if args.grayscale:
        aug_list.append(transforms.Grayscale())

    if args.grayscale:
        normalization_mean = [0.5]
        normalization_std = [0.5]
    else:
        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]

    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std)
    ])

    return aug_list


def get_data_augmentation(args: Namespace) -> Dict[str, Callable]:
    """
    Returns data augmentation transforms for training and validation sets.

    Parameters:
        args (Namespace): A namespace object containing the following attributes:
            crop_size (int): The size of the crop for the training and validation sets.
            val_resize (int): The target size for resizing the validation images.
                              The validation images will be resized to this size while maintaining their aspect ratio.
            interpolation (int): The interpolation method for resizing and cropping.
            hflip (bool): Whether to apply random horizontal flip to the training set.
            aug_type (str): The type of augmentation to apply to the training set.
                             Must be one of "trivial", "augmix", or "rand".

    Returns:
        Dict[str, Callable]: A dictionary of data augmentation transforms for the training and validation sets.

    Example:
        >>> import torchvision.transforms as transforms
        >>> args = Namespace(crop_size=224, interpolation=3, hflip=True, aug_type='augmix')
        >>> transforms_dict = get_data_augmentation(args)
        >>> train_transforms = transforms_dict['train']
        >>> val_transforms = transforms_dict['val']
        >>> print(train_transforms)
        Output: Composed transform with random resized crop, random horizontal flip, and AugMix augmentation
        >>> print(val_transforms)
        Output: Composed transform with resize, center crop, and normalization
    """
    def get_augmentation_by_type(aug_type: str) -> Callable:
        if aug_type == "trivial":
            return transforms.TrivialAugmentWide(num_magnitude_bins=args.mag_bins,
                                                 interpolation=f.InterpolationMode(args.interpolation))
        elif aug_type == "augmix":
            return transforms.AugMix(interpolation=f.InterpolationMode(args.interpolation))
        elif aug_type == "rand":
            return transforms.RandAugment(num_magnitude_bins=args.mag_bins,
                                          interpolation=f.InterpolationMode(args.interpolation))
        else:
            raise ValueError(f"Invalid augmentation type: '{aug_type}'")

    train_aug = [transforms.RandomResizedCrop(args.crop_size, interpolation=f.InterpolationMode(args.interpolation)),
                 transforms.RandomHorizontalFlip(args.hflip), get_augmentation_by_type(args.aug_type)]

    train_transform = transforms.Compose(apply_normalization(args, train_aug))

    val_aug = [
        transforms.Resize(args.val_resize, interpolation=f.InterpolationMode(args.interpolation)),
        transforms.CenterCrop(args.crop_size),
    ]
    val_transform = transforms.Compose(apply_normalization(args, val_aug))

    return {"train": train_transform, "val": val_transform}


def apply_mixup(images: Tensor, targets: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    Applies Mixup augmentation to input data.

    Parameters:
        images (Tensor): Input images.
        targets (Tensor): Corresponding targets.
        alpha (float, optional): Mixup parameter. Defaults to 1.0.

    Returns:
        Tuple[Tensor, Tensor, Tensor, float]: Mixed images, mixed labels (labels_a),
        original labels (labels_b), and mixup factor (lambda).
    Example:
         >>> import torch
         >>> images = torch.tensor([[1, 2, 3], [4, 5, 6]])
         >>> targets = torch.tensor([0, 1])
         >>> mixed_images, labels_a, labels_b, lam = apply_mixup(images, targets, alpha=0.5)
         >>> print(mixed_images)
         Output: tensor([[2.5000, 3.5000, 4.5000],
                         [2.0000, 2.5000, 3.0000]])
         >>> print(labels_a)
         Output: tensor([0, 1])
         >>> print(labels_b)
         Output: tensor([0, 1])
         >>> print(lam)
         Output: 0.5
    """
    beta_dist = Beta(alpha, alpha)
    lam = beta_dist.sample().item()

    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    mixed_images = lam * images + (1 - lam) * images[index, :]
    targets_a, targets_b = targets, targets[index]

    return mixed_images, targets_a, targets_b, lam


def apply_cutmix(images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor,
torch.Tensor, float]:
    """
    Applies CutMix augmentation to input data.

    Parameters:
        images (torch.Tensor): Input images.
        targets (torch.Tensor): Corresponding labels.
        alpha (float, optional): CutMix parameter. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: Mixed images, mixed labels (targets_a),
        original labels (targets_b), and mix factor (lambda).
    Example:
         >>> import torch
         >>> images = torch.tensor([[1, 2, 3], [4, 5, 6]])
         >>> targets = torch.tensor([0, 1])
         >>> mixed_images, targets_a, targets_b, lam = apply_cutmix(images, targets, alpha=0.5)
         >>> print(mixed_images)
         Output: tensor([[1, 2, 3],
                         [4, 5, 6]])
         >>> print(targets_a)
         Output: tensor([0, 1])
         >>> print(targets_b)
         Output: tensor([1, 0])
         >>> print(lam)
         Output: 0.5
    """
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    height, width = images.size(2), images.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = np.int(height * cut_ratio)
    cut_w = np.int(width * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    mixed_images = images.clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (width * height)

    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam


def to_channels_first(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor from channels-last format to channels-first format.

    Parameters:
        image (torch.Tensor): A 4-D or 3-D image tensor.

    Returns:
        torch.Tensor: The image tensor in channels-first format.

    Example:
         >>> import torch
         >>> image = torch.randn(32, 64, 64, 3)  # Example 4-D image tensor with channels last
         >>> converted_image = to_channels_first(image)
         >>> print(converted_image.shape)
         Output: torch.Size([32, 3, 64, 64])
    """
    return image.permute(0, 3, 1, 2) if image.dim() == 4 else image.permute(2, 0, 1)


def to_channels_last(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor from the channels-first format to the channels-last format.

    Parameters:
        image (torch.Tensor): A 4-D or 3-D image tensor in channels-first format.

    Returns:
        torch.Tensor: The image tensor in channels-last format.

    Example:
        >>> import torch
        >>> image = torch.randn(1, 3, 32, 32)  # Channels-first format
        >>> image_channels_last = to_channels_last(image)
        >>> print(image_channels_last.shape)
        Output: torch.Size([1, 32, 32, 3])
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

    Example:
        >>> args = Namespace(crop_size=224)
        >>> model_name = "resnet18"
        >>> checkpoint_path = "./best_model.pt"
        >>> num_classes = 10
        >>> convert_to_onnx(args, model_name, checkpoint_path, num_classes)
    """

    model = get_pretrained_model(args, model_name, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "n_averaged" in checkpoint["model"]:
        del checkpoint["model"]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["model"], "module.")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 1 if args.grayscale else 3, args.crop_size, args.crop_size, requires_grad=True)
    filename = os.path.join(os.path.dirname(checkpoint_path), "best_model.onnx")

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


def get_explanation_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns the transforms for data augmentation used for explaining the model.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: A tuple of two transforms representing the data augmentation
        transforms used for explanation and the inverse of those transforms.

    Example:
        >>> transform, inv_transform = get_explanation_transforms()
        >>> original_image = ...  # Load or create the original image
        >>> transformed_image = transform(original_image)
        >>> # Perform some model explanation using the transformed image
        >>> # If needed, revert the transformed image back to the original using the inverse transform
        >>> original_image_restored = inv_transform(transformed_image)
    """
    transform = transforms.Compose([
        transforms.Lambda(to_channels_first),
        transforms.Lambda(lambda image: image * (1 / 255)),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        transforms.Lambda(to_channels_last),
    ])

    inv_transform = transforms.Compose([
        transforms.Lambda(to_channels_first),
        transforms.Normalize(
            mean=(-1 * np.array(IMAGENET_DEFAULT_MEAN) / np.array(IMAGENET_DEFAULT_STD)).tolist(),
            std=(1 / np.array(IMAGENET_DEFAULT_STD)).tolist()
        ),
        transforms.Lambda(to_channels_last),
    ])

    return transform, inv_transform


def collate_fn(examples):
    """
    Collates a list of examples into batches by stacking pixel values of images and creating a tensor for labels.

    Parameters:
        examples (list): A list of examples, each containing a dictionary with "pixel_values" and "labels" keys.

    Returns:
        dict: A dictionary containing the batched pixel values and labels.

    Example:
        >>> examples = [{"pixel_values": torch.tensor([1, 2, 3]), "labels": 0},
                       {"pixel_values": torch.tensor([4, 5, 6]), "labels": 1},
                       {"pixel_values": torch.tensor([7, 8, 9]), "labels": 2}]
        >>> batch = collate_fn(examples)
        >>> print(batch)
        Output: {'pixel_values': tensor([[1, 2, 3],
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

    Example:
        >>> dataset = load_image_dataset('dataset')
        >>> classes = get_classes(dataset)
        >>> print(classes)
        Output: ['class1', 'class2', 'class3', ...]
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

    Example:
        >>> import argparse

        ### Loading models based on model size
        >>> args = argparse.Namespace(image_size=224, model_size="small")
        >>> get_matching_model_names(args)
        ['tf_efficientnet_b0_ns_small_224', 'tf_efficientnet_b1_ns_small_224', 'tf_efficientnet_b2_ns_small_224', ...]

        # Loading models based on specific submodule
        >>> args = argparse.Namespace(image_size=224, module="resnet")
        >>> get_matching_model_names(args)
        ['ecaresnet50d.miil_in1k', 'ecaresnet50t.a1_in1k', 'ecaresnet50t.a3_in1k', 'ecaresnet101d.miil_in1k', ...]
    """

    def filter_models(models: List[str], crop_size: int) -> List[str]:
        """
        Filter a list of model names based on crop size.

        This function iterates through the list of model names and removes
        those models that do not contain the given crop size in their name
        and have numeric values greater than the crop size in their suffixes.

        Args:
            models (List[str]): A list of model names.
            crop_size (int): The crop size to be checked against the models.

        Returns:
            List[str]: A filtered list of model names after the recurring filtering process.

        Example:
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
    else:
        model_names = timm.list_models(pretrained=True)

        models_to_remove = {
            "tiny": {"deit_tiny_distilled_patch16_224.fb_in1k", "swin_s3_tiny_224.ms_in1k"},
            "small": {"deit_small_distilled_patch16_224.fb_in1k"},
            "base": {"deit_base_distilled_patch16_224.fb_in1k", "vit_base_patch8_224.augreg2_in21k_ft_in1k"}
        }
        matching_models = [name for name in model_names if is_matching_model(name)]
        matching_models = [name for name in matching_models if name not in models_to_remove.get(args.model_size, set())]

    return matching_models


def prune_model(model: nn.Module, pruning_rate: float) -> List[Tuple[nn.Module, str]]:
    """
    Applies global unstructured pruning to the model.

    Parameters:
        model (nn.Module): The model to be pruned.
        pruning_rate (float): The fraction of weights to be pruned.

    Returns:
        List[Tuple[nn.Module, str]]: A list of tuples containing the pruned modules and parameter names.

    Example:
        >>> model = MyModel()
        >>> pruning_rate = 0.5
        >>> pruned_params = prune_model(model, pruning_rate)
        >>> print(pruned_params)
        Output: [(Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'weight'),
                 (Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'weight'),
                 (Linear(in_features=512, out_features=256, bias=True), 'weight'), ...]
    """

    parameters_to_prune = [
        (module, 'weight') for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

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
        parameters_to_prune (List[Tuple[nn.Module, str]]): List of module and parameter names to remove pruning
        re-parametrization.

    Example:
         >>> model = MyModel()
         >>> parameters_to_prune = prune_model(model, pruning_rate=0.2)
         >>> remove_pruning_reparam(parameters_to_prune)
    """
    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)


def get_pretrained_model(args: Namespace, model_name: str, num_classes: int) -> nn.Module:
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

    Example:
        >>> args = Namespace(feat_extract=True, dropout=0.5, grayscale=False)
        >>> model = get_pretrained_model(args, "tf_efficientnet_b0_ns", num_classes=10)
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        scriptable=True,
        exportable=True,
        drop_rate=args.dropout,
        in_chans=1 if args.grayscale else 3,
        num_classes=num_classes
    )

    if args.feat_extract:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model


# noinspection PyTypeChecker
def calculate_class_weights(data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Returns the class weights for the given data loader.

    The class weights are calculated as the inverse frequency of each class in the dataset.

    Parameters:
        data_loader (torch.utils.data.DataLoader): A PyTorch data loader.

    Returns:
        torch.Tensor: A tensor of class weights.

    Example:
        # Assuming you have already loaded the image dataset and set up the data loader:
        >>> train_dataset = load_image_dataset(args.dataset)["train"]
        >>> data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Call the function to calculate class weights
        >>> class_weights = calculate_class_weights(data_loader)

        # Now you can use the calculated class weights for your training or evaluation
        # For example, you might use these class weights in your loss function to handle class imbalance.
    """
    targets = data_loader.dataset[:]["labels"]
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    num_classes = len(get_classes(data_loader.dataset))
    class_weights = torch.tensor(total_samples / (num_classes * class_counts), dtype=torch.float32)
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

    Example:
        >>> checkpoint_paths = ["checkpoint1.pth", "checkpoint2.pth", "checkpoint3.pth"]
        >>> averaged_params = average_checkpoints(checkpoint_paths)
        >>> print(averaged_params)  # Display the averaged parameters
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


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Returns a list of trainable parameters in the given model.

    Parameters:
        model (nn.Module): A PyTorch neural network model.

    Returns:
        List[nn.Parameter]: A list of trainable parameters in the model.

    Example:
        >>> model = nn.Linear(10, 5)
        >>> trainable_params = get_trainable_params(model)
        >>> len(trainable_params)
        2
    """
    return list(filter(lambda param: param.requires_grad, model.parameters()))


def get_optimizer(args: Namespace, params: List[nn.Parameter]) -> optim.Optimizer:
    """
    Returns an optimizer object based on the provided optimization algorithm name.

    Parameters:
        args (Namespace): A namespace object containing the following attributes:
            - opt_name (str): The name of the optimization algorithm.
            - lr (float): The learning rate for the optimizer.
            - wd (float): The weight decay for the optimizer.
        params (List[nn.Parameter]): A list of parameters for the optimizer.

    Returns:
        optim.Optimizer: An optimizer object of the specified type.

    Example:
        >>> args = Namespace(opt_name="sgd", lr=0.01, wd=0.0001)
        >>> model = torch.nn.Linear(10, 10)
        >>> params = get_trainable_parameters(model)
        >>> optimizer = get_optimizer(args, params)
    """
    optimizer = create_optimizer_v2(model_or_params=params, opt=args.opt_name, lr=args.lr, weight_decay=args.wd)
    return optimizer


def get_lr_scheduler(args: Namespace, optimizer: optim.Optimizer, num_iters: int) -> Union[lr_scheduler.SequentialLR,
                                                                                         lr_scheduler.LRScheduler,
                                                                                         None]:
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

    Example:
        # Obtain a learning rate scheduler based on the provided args and optimizer, and use it during training:

        >>> scheduler = get_lr_scheduler(args, optimizer, num_iters)
        >>> if scheduler is not None:
        >>>     for epoch in range(args.epochs):
        >>>         scheduler.step()
        >>>         train_one_epoch()
        >>>         evaluate()
    """
    if args.sched_name == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.sched_name == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                                                   eta_min=args.eta_min)
    elif args.sched_name == "cosine_wr":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, eta_min=args.eta_min)
    elif args.sched_name == "one_cycle":
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs,
                                            steps_per_epoch=num_iters)
    else:
        return None

    if args.warmup_epochs > 0:
        warmup_lr = lr_scheduler.LinearLR(optimizer, start_factor=args.warmup_decay, total_iters=args.warmup_epochs)
        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr, scheduler],
                                              milestones=[args.warmup_epochs])

    return scheduler


# adapted from https://github.com/pytorch/vision/blob/a5035df501747c8fc2cd7f6c1a41c44ce6934db3/references
# /classification/utils.py#L159
class ExponentialMovingAverage(swa_utils.AveragedModel):
    """
    Exponential Moving Average (EMA) implementation for model parameters.

    Parameters:
        model (torch.nn.Module): The model to apply EMA to.
        decay (float): The decay factor for EMA.
        device (str, optional): The device to use for EMA. Defaults to "cpu".

    Example:
         # Create an Exponential Moving Average object for a model with a decay factor of 0.9, and update the
         # parameters:

        >>> model = MyModel()
        >>> ema = ExponentialMovingAverage(model, decay=0.9)
        >>> ema.update_parameters(model)
    """

    def __init__(self, model: nn.Module, decay: float, device: str = "cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


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
                    shutil.copy(file, os.path.join(self.img_dest, dir_name))
                    break


def create_train_val_test_splits(img_src: str, img_dest: str, ratio: tuple) -> None:
    """
    Split images from `img_src` directory into train, validation, and test sets and save them in `img_dest`
    directory. This will save the images in the appropriate directories based on the train-val-test split ratio.

    Parameters:
        img_src (str): The source directory containing the images to be split.
        img_dest (str): The destination directory where the split images will be saved.
        ratio (tuple): The train, val, test splits. E.g (0.8, 0.1, 0.1)

    Example:
        # Split images from "data/images" into train, validation, and test sets with a split ratio of (0.8, 0.1, 0.1)
        # and save them in the "data/splits" directory:

        >>> create_train_val_test_splits("data/images", "data/splits", ratio=(0.8, 0.1, 0.1))
    """
    splitfolders.ratio(img_src, output=img_dest, seed=333777999, ratio=ratio)
