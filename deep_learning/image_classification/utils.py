import os
import re
import json
import random
import shutil
import logging

from glob import glob
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import timm
import torch
import numpy as np
import splitfolders
import torch.nn.utils.prune as prune
import torch.optim.lr_scheduler as lr_scheduler

from torch.optim import swa_utils
from torch import nn, optim, Tensor
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
        print_header("Hello", "World", 2023)
        Output: Hello World 2023
    """
    message = " ".join(map(str, args))
    print(message)


def print_heading(message: str) -> None:
    """
    Print a formatted heading with the given message.

    Parameters:
        message (str): The message to be included in the heading.

    Example:
        print_heading("Welcome to the Chatbot")
        Output:
        =========================
        Welcome to the Chatbot
        =========================
    """
    line = "=" * len(message) + "=="
    print_header(line)
    print_header(message)
    print_header(line)


def get_model_run_id(run_ids: Dict[str, str], model_name: str) -> Optional[str]:
    """
    Get the run ID of a specific model from a dictionary of run IDs.

    Parameters:
        run_ids (Dict[str, str]): A dictionary mapping model names to run IDs.
        model_name (str): The name of the model for which to retrieve the run ID.

    Returns:
        Optional[str]: The run ID of the specified model, or None if the model is not in the dictionary.

    Example:
        run_ids = {"model1": "1234", "model2": "5678", "model3": "9012"}  # Example dictionary of run IDs
        model_name = "model2"  # Example model name to retrieve the run ID
        run_id = get_model_run_id(run_ids, model_name)
        print(run_id)
        Output: 5678
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
        dictionary = {"key": "value"}  # Example dictionary object to be written
        file_path = "data.json"  # Example file path
        write_dictionary_to_json(dictionary, file_path)
    """
    with open(file_path, "w") as file_out:
        json.dump(dictionary, file_out, indent=4)


def append_dictionary_to_json_file(new_dict: Dict, file_path: str) -> None:
    """
    Append a dictionary to a JSON file at the given file path.
    If the file does not exist, it will be created.

    Parameters:
        new_dict (Dict): The dictionary to be appended to the JSON file.
        file_path (str): The path to the JSON file.

    Returns:
        None

    Example:
        new_dict = {"key": "value"}  # Example dictionary to be appended
        file_path = "data.json"  # Example file path
        append_dictionary_to_json_file(new_dict, file_path)
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
        file_path = "data.json"  # Example file path
        data = read_json_file(file_path)
        print(data)
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
        file_path = "data.jsonl"  # Example file path
        data = read_json_lines_file(file_path)
        print(data)
    """
    with open(file_path, "r") as file_in:
        data = [json.loads(line) for line in file_in]
        return data


def set_random_seeds(seed: int) -> None:
    """
    Sets the seed for the random number generators in NumPy, Python's random module, and PyTorch.

    Parameters:
        seed (int): The seed to be set.

    Returns:
        None

    Example:
        seed_value = 123  # Example seed value
        set_random_seeds(seed_value)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_seed_for_worker(worker_id: Optional[int]) -> Optional[int]:
    """
    Sets the seed for NumPy and Python's random module for the given worker.
    If no worker ID is provided, uses the initial seed for PyTorch and returns None.

    Parameters:
        worker_id (Optional[int]): The ID of the worker.

    Returns:
        Optional[int]: The seed used for the worker, or None if no worker ID was provided.

    Example:
        worker_id = 1  # Example worker ID
        seed = set_seed_for_worker(worker_id)
        print(seed)  # Seed used for the worker
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


def apply_fgsm_attack(model: nn.Module, loss: Tensor, images: Tensor, epsilon: float) -> Tensor:
    """
    Applies the Fast Gradient Sign Method (FGSM) attack to the input image.

    Parameters:
        model (nn.Module): The model used for generating gradients.
        loss (Tensor): The loss value used for calculating gradients.
        images (Tensor): The input image to be perturbed.
        epsilon (float): The perturbation magnitude.

    Returns:
        Tensor: The perturbed image.

    Example:
        import torch

        model = MyModel()  # Example model
        image = torch.randn(3, 224, 224)  # Example input image
        epsilon = 0.03  # Perturbation magnitude
        loss = calculate_loss(model, image)  # Example loss calculation
        perturbed_image = apply_fgsm_attack(model, loss, image, epsilon)
        print(perturbed_image)  # Perturbed image after applying FGSM attack
    """
    model.eval()
    images.requires_grad_(True)

    model.zero_grad()
    loss.backward()

    gradients = images.grad.data
    image_perturbed = images + epsilon * torch.sign(gradients)
    image_perturbed = torch.clamp(image_perturbed, min=0, max=1)
    return image_perturbed


def apply_normalization(aug_list: List, args: Namespace) -> List:
    """
    Apply normalization to the augmentation list based on grayscale conversion.

    Parameters:
        aug_list (List[transforms.Transform]): The list of transformation functions for data augmentation.
        args (Namespace): A namespace object containing the following attributes:
            - gray (bool): Whether to convert the images to grayscale.

    Returns:
        List: The updated list of transformation functions with normalization applied.

    Example:
        args = Namespace(gray=True)
        aug_list = [transforms.RandomResizedCrop(224), transforms.ToTensor()]
        updated_aug_list = apply_normalization(aug_list, args)
    """
    if args.gray:
        aug_list.append(transforms.Grayscale())

    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) if args.gray else transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
    ])

    return aug_list


def get_data_augmentation(args: Namespace) -> Dict[str, Callable]:
    """
    Returns data augmentation transforms for training and validation sets.


    Parameters:
        args: A namespace object containing the following attributes:
            - crop_size (int): The size of the crop for the training and validation sets.
            - interpolation (int): The interpolation method for resizing and cropping.
            - hflip (bool): Whether to apply random horizontal flip to the training set.
            - aug_type (str): The type of augmentation to apply to the training set.
                              Must be one of "trivial", "augmix", or "rand".

    Returns:
        Dict[str, Callable]: A dictionary of data augmentation transforms for the training and validation sets.

    Example:
         import torchvision.transforms as transforms
         args = Namespace(crop_size=224, interpolation=3, hflip=True, aug_type='augmix')
         transforms_dict = get_data_augmentation(args)
         train_transforms = transforms_dict['train']
         val_transforms = transforms_dict['val']
         print(train_transforms)
         # Composed transform with random resized crop, random horizontal flip, and AugMix augmentation
         print(val_transforms)  # Composed transform with resize, center crop, and normalization
    """
    train_aug = [
        transforms.RandomResizedCrop(args.crop_size, interpolation=f.InterpolationMode(args.interpolation)),
        transforms.RandomHorizontalFlip(args.hflip)
    ]
    if args.aug_type == "trivial":
        train_aug.append(transforms.TrivialAugmentWide(num_magnitude_bins=args.mag_bins,
                                                       interpolation=f.InterpolationMode(args.interpolation)))
    elif args.aug_type == "augmix":
        train_aug.append(transforms.AugMix(interpolation=f.InterpolationMode(args.interpolation)))
    elif args.aug_type == "rand":
        train_aug.append(transforms.RandAugment(num_magnitude_bins=args.mag_bins,
                                                interpolation=f.InterpolationMode(args.interpolation)))
    train_transform = transforms.Compose(apply_normalization(train_aug, args))

    val_aug = [
        transforms.Resize(args.val_resize, interpolation=f.InterpolationMode(args.interpolation)),
        transforms.CenterCrop(args.crop_size),
    ]
    val_transform = transforms.Compose(apply_normalization(val_aug, args))

    return {"train": train_transform, "val": val_transform}


def apply_mixup(images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor,
torch.Tensor, float]:
    """
    Applies Mixup augmentation to input data.

    Parameters:
        images (torch.Tensor): Input images.
        targets (torch.Tensor): Corresponding targets.
        alpha (float, optional): Mixup parameter. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: Mixed images, mixed labels (labels_a),
        original labels (labels_b), and mixup factor (lambda).
    Example:
         import torch
         images = torch.tensor([[1, 2, 3], [4, 5, 6]])
         targets = torch.tensor([0, 1])
         mixed_images, labels_a, labels_b, lam = mixup_data(images, targets, alpha=0.5)
         print(mixed_images)  # [[2 4 4], [5 8 9]]
         print(labels_a)  # [1.0, 0.5]
         print(labels_b)  # [0.0, 0.5]
         print(lam)  # 0.5
    """
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
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
         import torch
         images = torch.tensor([[1, 2, 3], [4, 5, 6]])
         targets = torch.tensor([0, 1])
         mixed_images, targets_a, targets_b, lam = cutmix_data(images, targets, alpha=0.5)
         print(mixed_images)  # [[1 2 6], [4 5 3]]
         print(targets_a)  # [0, 1]
         print(targets_b)  # [1, 0]
         print(lam)  # 0.5
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
         import torch
         image = torch.randn(32, 64, 64, 3)  # Example 4-D image tensor with channels last
         converted_image = convert_to_channels_first(image)
         print(converted_image.shape)
        torch.Size([32, 3, 64, 64])
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
        import torch

        image = torch.randn(1, 3, 32, 32)  # Channels-first format
        image_channels_last = convert_to_channels_last(image)
        print(image_channels_last.shape)
        # torch.Size([1, 32, 32, 3])
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
            - crop_size (int, optional): The size of the crop for the inference dataset/image. Default: None.
        model_name (str): The name of the model.
        checkpoint_path (str): The path to the PyTorch checkpoint.
        num_classes (int): The number of classes in the dataset.

    Example:
        convert_to_onnx(args, "resnet18", "./best_model.pt", 10)
    """

    model = get_pretrained_model(args, model_name, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "n_averaged" in checkpoint["model"]:
        del checkpoint["model"]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["model"], "module.")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 1 if args.gray else 3, args.crop_size, args.crop_size, requires_grad=True)
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
        transform, inv_transform = get_explain_data_aug()
        transformed_image = transform(original_image)
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


def get_classes(dataset_dir: str) -> List[str]:
    """
    Get a list of the classes in a dataset.

    Parameters:
        dataset_dir (str): Directory of the dataset.

    Returns:
        List[str]: A sorted list of the classes in the dataset.

    Example:
        classes = get_classes('path/to/dataset')
        print(classes)
        # ['class1', 'class2', 'class3', ...]
    """
    class_dirs = glob(os.path.join(dataset_dir, "train", "*"))
    classes = [os.path.basename(class_dir) for class_dir in class_dirs]
    classes.sort()

    return classes


def get_matching_model_names(image_size: int, model_size: str) -> List[str]:
    """
    Get a list of model names that match the given image size and model size.

    Parameters:
        image_size (int): Image size the models should be trained on.
        model_size (str): Size of the model (e.g., "tiny", "small", etc.).

    Returns:
        List[str]: A list of model names that can be used for training.

    Example:
        get_matching_model_names(224, "small")
        # ['tf_efficientnet_b0_ns_small_224', 'tf_efficientnet_b1_ns_small_224', 'tf_efficientnet_b2_ns_small_224', ...]
    """
    model_names = timm.list_models(pretrained=True)

    matching_models = [name for name in model_names if str(image_size) in name and model_size in name]

    training_models = [
        name for name in matching_models if isinstance(list(timm.create_model(name).named_modules())[-1][1],
                                                       torch.nn.Linear)
    ]

    return training_models


def prune_model(model: nn.Module, pruning_rate: float) -> List[Tuple[nn.Module, str]]:
    """
    Applies global unstructured pruning to the model.

    Parameters:
        model (nn.Module): The model to be pruned.
        pruning_rate (float): The fraction of weights to be pruned.

    Returns:
        List[Tuple[nn.Module, str]]: A list of tuples containing the pruned modules and parameter names.

    Example:
        model = MyModel()
        pruning_rate = 0.5
        pruned_params = prune_model(model, pruning_rate)
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
         model = MyModel()
         parameters_to_prune = prune_model(model, pruning_rate=0.2)
         remove_pruning_reparam(parameters_to_prune)
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
         model = get_pretrained_model(args, "tf_efficientnet_b0_ns", num_classes=10)
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        scriptable=True,
        exportable=True,
        drop_rate=args.dropout,
        in_chans=1 if args.gray else 3
    )

    if args.feat_extract:
        for param in model.parameters():
            param.requires_grad = False

    if isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes, bias=True)
    elif hasattr(model.head, "fc") and isinstance(model.head.fc, nn.Linear):
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes, bias=True)

    if hasattr(model, "head_dist") and isinstance(model.head_dist, nn.Linear):
        model.head_dist = nn.Linear(model.head_dist.in_features, num_classes, bias=True)

    return model


def calculate_class_weights(data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Returns the class weights for the given data loader.

    The class weights are calculated as the inverse frequency of each class in the dataset.

    Parameters:
        data_loader (torch.utils.data.DataLoader): A PyTorch data loader.

    Returns:
        torch.Tensor: A tensor of class weights.

    Example:
         from torch.utils.data import DataLoader
         from torchvision.datasets import ImageFolder
         from torchvision import transforms

         dataset = ImageFolder("path/to/dataset", transform=transforms.ToTensor())
         data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
         class_weights = calculate_class_weights(data_loader)
    """
    targets = data_loader.dataset.targets
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    num_classes = len(data_loader.dataset.classes)
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
        checkpoint_paths = ["checkpoint1.pth", "checkpoint2.pth", "checkpoint3.pth"]
        averaged_params = average_checkpoints(checkpoint_paths)
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
         model = nn.Linear(10, 5)
         trainable_params = get_trainable_params(model)
         len(trainable_params)
         # 2
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
         args = Namespace(opt_name="sgd", lr=0.01, wd=0.0001)
         model = torch.nn.Linear(10, 10)
         params = get_trainable_parameters(model)
         optimizer = get_optimizer(args, params)
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
        scheduler = get_lr_scheduler(args, optimizer)
        if scheduler is not None:
            for epoch in range(args.epochs):
                scheduler.step()
                train_one_epoch()
                evaluate()
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
         model = MyModel()
         ema = ExponentialMovingAverage(model, decay=0.9)
         ema.update_parameters(model)
    """

    def __init__(self, model: nn.Module, decay: float, device: str = "cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def get_logger(logger_name: str, log_file: str, log_level: int = logging.DEBUG,
               console_level: int = logging.INFO) -> logging.Logger:
    """
    Creates a logger with a specified name, log file, and log level.
    The logger logs messages to a file and to the console.

    Parameters:
        logger_name (str): The name of the logger.
        log_file (str): The file path of the log file.
        log_level (int, optional): The log level for the file handler. Defaults to logging.DEBUG.
        console_level (int, optional): The log level for the console handler. Defaults to logging.INFO.

    Returns:
        logging.Logger: A logger object.

    Example:
        logger = get_logger("my_logger", "my_log.log")
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


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
         create_train_val_test_splits("data/images", "data/splits", ratio=(0.8, 0.1, 0.1))
    """
    splitfolders.ratio(img_src, output=img_dest, seed=333777999, ratio=ratio)
