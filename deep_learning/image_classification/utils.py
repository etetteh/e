import json
import logging
import os
import random
import re
import shutil
import sys
from glob import glob
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import splitfolders
import timm
import torch
from timm.optim import create_optimizer_v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn, optim, Tensor
import torch.nn.utils.prune as prune
from torchvision import transforms
from torchvision.transforms import functional as f


def header(*args: Union[str, int, float]):
    """
    Print one or more arguments to standard output as a single line of text.

    Parameters:
        - args: The arguments to be printed.

    Example:
         header("Hello", "World", 2023)
        Hello World 2023
    """
    msg = " ".join(map(str, args))
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def heading(message: str):
    """
    Print a formatted heading with a given message.

    Parameters:
         - message (str): The message to be included in the heading.

    Example:
         heading("Welcome to the Chatbot")
        ==========================
        Welcome to the Chatbot
        ==========================
    """
    header("=" * 99)
    header(message)
    header("=" * 99)


def get_run_id(run_ids: Dict[str, str], model_name: str) -> Optional[str]:
    """
    Get the run ID of a specific model from a dictionary of run IDs.

    Parameters:
        - run_ids (Dict[str, str]): A dictionary mapping model names to run IDs.
        - model_name (str): The name of the model for which to retrieve the run ID.

    Returns:
        - Optional[str]: The run ID of the specified model, or None if the model is not in the dictionary.

    Example:
         run_ids = {"model1": "1234", "model2": "5678", "model3": "9012"}  # Example dictionary of run IDs
         model_name = "model2"  # Example model name to retrieve the run ID
         run_id = get_run_id(run_ids, model_name)
         print(run_id)
        5678
    """
    try:
        run_id = run_ids[model_name]
    except KeyError:
        return None
    return run_id


def write_json_file(dict_obj: Dict, file_path: str):
    """
    Write a dictionary object to a JSON file at the given file path.
    If the file already exists, its content will be overwritten.

    Parameters:
        - dict_obj (Dict): The dictionary object to be written to the JSON file.
        - file_path (str): The path to the JSON file.

    Example:
         dict_obj = {"key": "value"}  # Example dictionary object to be written
         file_path = "data.json"  # Example file path
         write_json_file(dict_obj, file_path)
    """
    with open(file_path, "w") as file_out:
        json.dump(dict_obj, file_out, indent=4)


def append_dict_to_json_file(new_dict: Dict, file_path: str):
    """
    Add a dictionary to a JSON file at the given file path.
    If the file does not exist, it will be created.

    Parameters:
        - new_dict (Dict): The dictionary to be added to the JSON file.
        - file_path (str): The path to the JSON file.

    Example:
         new_dict = {"key": "value"}  # Example dictionary to be appended
         file_path = "data.json"  # Example file path
         append_dict_to_json_file(new_dict, file_path)
    """
    try:
        data = load_json_file(file_path)
    except FileNotFoundError:
        data = {}

    data.update(new_dict)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_json_file(file_path: str) -> Union[dict, list]:
    """
    Load a JSON file from the given file path and return the data as a dictionary.

    Parameters:
        - file_path (str): The path to the JSON file.

    Returns:
        - Union[dict, list]: The data contained in the JSON file as a dictionary.

    Example:
         file_path = "data.json"  # Example file path
         data = load_json_file(file_path)
         print(data)
    """
    with open(file_path, "r") as file:
        return json.load(file)


def load_json_lines_file(file_path: str) -> Union[dict, list]:
    """
    Load a JSON Lines file from the given file path and return the data as a list.

    Parameters:
        - file_path (str): The path to the JSON Lines file.

    Returns:
        - Union[dict, list]: The data contained in the JSON file as a list.

    Example:
         file_path = "data.jsonl"  # Example file path
         data = load_json_lines_file(file_path)
         print(data)
    """
    with open(file_path, "r") as file_in:
        return [json.loads(line) for line in file_in]


def set_seed_for_all(seed: int) -> None:
    """
    Sets the seed for NumPy, Python's random module, and PyTorch.

    Parameters:
        - seed (int): The seed to be set.

    Example:
         seed_value = 123  # Example seed value
         set_seed_for_all(seed_value)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_seed_for_worker(worker_id: Optional[int]) -> Optional[int]:
    """
    Sets the seed for NumPy and Python's random module for the given worker.
    If no worker ID is provided, uses the initial seed for PyTorch and returns None.

    Parameters:
        - worker_id (Optional[int]): The ID of the worker.

    Returns:
        - Optional[int]: The seed used for the worker, or None if no worker ID was provided.

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
        np.random.seed(torch.initial_seed() % 2 ** 32)
        random.seed(torch.initial_seed() % 2 ** 32)
        return None


def fgsm_attack(image: Tensor, epsilon: float, image_grad: Tensor) -> Tensor:
    """
    Applies Fast Gradient Sign Method (FGSM) attack to the input image.

    Parameters:
        image (Tensor): The input image to be perturbed.
        epsilon (float): The perturbation magnitude.
        image_grad (Tensor): The gradient of the loss function with respect to the input image.

    Returns:
        Tensor: The perturbed image.

    Example:
         import torch
         image = torch.randn(3, 224, 224)  # Example input image
         epsilon = 0.03  # Perturbation magnitude
         image_grad = torch.randn(3, 224, 224)  # Gradient of the loss function with respect to the image
         perturbed_image = fgsm_attack(image, epsilon, image_grad)
         print(perturbed_image)  # Perturbed image after applying FGSM attack
    """
    image_grad_sign = image_grad.sign()
    image_perturbed = image + epsilon * image_grad_sign
    image_perturbed = torch.clamp(image_perturbed, min=0, max=1)
    return image_perturbed


def get_data_augmentation(args) -> Dict[str, Callable]:
    """
    Returns data augmentation transforms for training and validation sets.
    The training set transform includes random resized crop, random horizontal flip,
    and one of three augmentations (trivial, augmix, or rand). The validation set
    transform includes resize, center crop, and normalization.

    Parameters:
        - args: A namespace object containing the following attributes:
            - crop_size (int): The size of the crop for the training and validation sets.
            - interpolation (int): The interpolation method for resizing and cropping.
            - hflip (bool): Whether to apply random horizontal flip to the training set.
            - aug_type (str): The type of augmentation to apply to the training set.
                              Must be one of "trivial", "augmix", or "rand".

    Returns:
        - Dict[str, Callable]: A dictionary of data augmentation transforms for the training and validation sets.

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
    train_aug.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    train_transform = transforms.Compose(train_aug)

    val_transform = transforms.Compose([
        transforms.Resize(args.val_resize, interpolation=f.InterpolationMode(args.interpolation)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    return {"train": train_transform, "val": val_transform}


def mixup_data(images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor,
torch.Tensor, float]:
    """
    Applies Mixup augmentation to input data.

    Parameters:
        - images (torch.Tensor): Input images.
        - targets (torch.Tensor): Corresponding targets.
        - alpha (float, optional): Mixup parameter. Defaults to 1.0.

    Returns:
        - tuple: Mixed images, mixed labels (labels_a), original labels (labels_b), and mixup factor (lambda).

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


def cutmix_data(images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor,
torch.Tensor, float]:
    """
    Applies CutMix augmentation to input data.

    Parameters:
        - images (torch.Tensor): Input images.
        - targets (torch.Tensor): Corresponding labels.
        - alpha (float, optional): CutMix parameter. Defaults to 1.0.

    Returns:
        - tuple: Mixed images, mixed labels (targets_a), original labels (targets_b), and mix factor (lambda).

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


def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a torch tensor to a NumPy ndarray.

    Parameters:
        - tensor (torch.Tensor): The torch tensor to be converted.

    Returns:
        - np.ndarray: The NumPy ndarray converted from the tensor.

    Example:
         import torch
         tensor = torch.tensor([1, 2, 3])
         numpy_array = convert_tensor_to_numpy(tensor)
         print(numpy_array)  # [1 2 3]
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def convert_to_channels_first(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor from channels-last format to channels-first format.

    Parameters:
        - image (torch.Tensor): A 4-D or 3-D image tensor.

    Returns:
        - torch.Tensor: The image tensor in channels-first format.

    Example:
         import torch
         image = torch.randn(32, 64, 64, 3)  # Example 4-D image tensor with channels last
         converted_image = convert_to_channels_first(image)
         print(converted_image.shape)
        torch.Size([32, 3, 64, 64])
    """
    return image.permute(0, 3, 1, 2) if image.dim() == 4 else image.permute(2, 0, 1)


def convert_to_channels_last(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor from channels-first format to channels-last format.

    Parameters:
        - image (torch.Tensor): A 4-D or 3-D image tensor.

    Returns:
        - torch.Tensor: The image tensor in channels-last format.

    Example:
         image = torch.randn(1, 3, 32, 32)  # Channels-first format
         image_channels_last = convert_to_channels_last(image)
         print(image_channels_last.shape)
        torch.Size([1, 32, 32, 3])
    """
    return image.permute(0, 2, 3, 1) if image.dim() == 4 else image.permute(1, 2, 0)


def convert_to_onnx(model_name: str, checkpoint_path: str, num_classes: int, dropout: float, crop_size: int) -> None:
    """
    Convert a PyTorch model to ONNX format.

    Parameters:
        - model_name (str): The name of the model.
        - checkpoint_path (str): The path to the PyTorch checkpoint.
        - num_classes (int): The number of classes in the dataset.
        - dropout (float): The dropout rate to be used in the model.
        - crop_size (int): The size of the crop for inference dataset/image.

    Example:
         convert_to_onnx("resnet18", "./best_model.pt", 10, 0.2)

    """

    model = get_model(model_name, num_classes, dropout)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "n_averaged" in checkpoint["model"]:
        del checkpoint["model"]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["model"], "module.")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, crop_size, crop_size, requires_grad=True)
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


def get_explain_data_aug() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns the transforms for data augmentation used for explaining the model.

    Returns:
        - A tuple of two transforms representing the data augmentation transforms
          used for explanation and the inverse of those transforms.

    Example:
         transform, inv_transform = get_explain_data_aug()
         transformed_image = transform(original_image)
    """
    transform = transforms.Compose([transforms.Lambda(convert_to_channels_first),
                                    transforms.Lambda(lambda image: image * (1 / 255)),
                                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                                    transforms.Lambda(convert_to_channels_last),
                                    ])

    inv_transform = transforms.Compose([
        transforms.Lambda(convert_to_channels_first),
        transforms.Normalize(
            mean=(-1 * np.array(IMAGENET_DEFAULT_MEAN) / np.array(IMAGENET_DEFAULT_STD)).tolist(),
            std=(1 / np.array(IMAGENET_DEFAULT_STD)).tolist()
        ),
        transforms.Lambda(convert_to_channels_last),
    ])

    return transform, inv_transform


def get_classes(dataset_dir: str) -> List[str]:
    """
    Get a list of the classes in a dataset.

    Parameters:
        - dataset_dir: Directory of the dataset.

    Returns:
        - A sorted list of the classes in the dataset.

    Example:
         classes = get_classes('path/to/dataset')
         print(classes)
        ['class1', 'class2', 'class3', ...]
    """
    class_dirs = glob(os.path.join(dataset_dir, "train/*"))
    classes = [os.path.basename(class_dir) for class_dir in class_dirs]
    classes.sort()

    return classes


def get_model_names(image_size: int, model_size: str) -> List[str]:
    """
    Get a list of model names that match the given image size and model size.

    Parameters:
        - image_size: Image size the models should be trained on.
        - model_size: Size of the model (e.g. "tiny", "small", etc.)

    Returns:
        - A list of model names that can be used for training.

    Example:
         get_model_names(224, "small")
        ['tf_efficientnet_b0_ns_small_224', 'tf_efficientnet_b1_ns_small_224', 'tf_efficientnet_b2_ns_small_224', ...]
    """
    model_names = timm.list_pretrained()

    model_names = [m for m in model_names if str(image_size) in m and model_size in m]

    training_models = [m for m in model_names if
                       isinstance(list(timm.create_model(m).named_modules())[-1][1], torch.nn.Linear)]

    return training_models


def prune_model(model: nn.Module, pruning_rate: float) -> List[Tuple[nn.Module, str]]:
    """
    Applies global unstructured pruning to the model.

    Parameters:
        - model (nn.Module): The model to be pruned.
        - pruning_rate (float): The fraction of weights to be pruned.

    Returns:
        List[Tuple[nn.Module, str]]: A list of tuples containing the pruned modules and parameter names.

    Example:
         model = MyModel()
         pruning_rate = 0.5
         pruned_params = prune_model(model, pruning_rate)
    """

    parameters_to_prune = [(module, 'weight') for module in model.modules() if
                           isinstance(module, torch.nn.Conv2d) or
                           isinstance(module, torch.nn.Linear)
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
        - parameters_to_prune (List[Tuple[nn.Module, str]]): List of module and parameter names to remove pruning
          re-parametrization.

    Returns:
        None

    Example:
         model = MyModel()
         parameters_to_prune = prune_model(model, pruning_rate=0.2)
         remove_pruning_reparam(parameters_to_prune)
    """
    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)


def get_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    """
    Returns a pretrained model with a new head and the model name.

    The head of the model is replaced with a new linear layer with the given
    number of classes.

    Parameters:
        - dropout (float): The dropout rate
        - model_name (str): The name of the model to be created using the `timm` library.
        - num_classes (int): The number of classes for the new head of the model.

    Returns:
        - Tuple[nn.Module, str]: A tuple containing the modified model and the model name.

    Example:
         model = get_model("tf_efficientnet_b0_ns", num_classes=10, dropout=0.1)
    """
    model = timm.create_model(model_name,
                              pretrained=True,
                              scriptable=True,
                              exportable=True,
                              drop_rate=dropout
                              )
    freeze_params(model)

    if hasattr(model.head, "fc"):
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, num_classes, bias=True)

        if hasattr(model, "head_dist"):
            model.head_dist = nn.Linear(num_ftrs, num_classes, bias=True)

    if hasattr(model.head, "in_features") and not hasattr(model.head, "fc"):
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes, bias=True)

        if hasattr(model, "head_dist"):
            model.head_dist = nn.Linear(num_ftrs, num_classes, bias=True)

    return model


def freeze_params(model: nn.Module) -> None:
    """
    Freezes the parameters in the given model.

    Parameters:
        - model (nn.Module): A PyTorch neural network model.

    Example:
         model = MyModel()
         freeze_params(model)
        # Now all the parameters in `model` are frozen and won't be updated during training.
    """
    for param in model.parameters():
        param.requires_grad = False


def get_class_weights(data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Returns the class weights for the given data loader.

    The class weights are calculated as the inverse frequency of each class in the dataset.

    Parameters:
        - data_loader (torch.utils.data.DataLoader): A PyTorch data loader.

    Returns:
        - torch.Tensor: A tensor of class weights.

    Example:
         from torch.utils.data import DataLoader
         from torchvision.datasets import ImageFolder
         from torchvision import transforms

         dataset = ImageFolder("path/to/dataset", transform=transforms.ToTensor())
         data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
         class_weights = get_class_weights(data_loader)
    """
    targets = data_loader.dataset.targets
    class_counts = np.bincount(targets)
    class_weights = len(targets) / (len(data_loader.dataset.classes) * class_counts)
    return torch.Tensor(class_weights)


# adapted from https://github.com/pytorch/vision/blob/a5035df501747c8fc2cd7f6c1a41c44ce6934db3/references
# /classification/utils.py#L272
def average_checkpoints(inputs):
    """
    Averages the parameters of multiple checkpoints.

    Parameters:
       - inputs (List[str]): List of file paths to the input checkpoint files.

    Returns:
       - OrderedDict: Averaged parameters in the form of an ordered dictionary.

    Raises:
       KeyError: If the checkpoints have different sets of parameters.

    Example:
        inputs = ["checkpoint1.pth", "checkpoint2.pth", "checkpoint3.pth"]
        averaged_params = average_checkpoints(inputs)
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f_in:
            state = torch.load(
                f_in,
                map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")),
            )
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f_in}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Returns a list of trainable parameters in the given model.

    Parameters:
        - model (nn.Module): A PyTorch neural network model.

    Returns:
        - List[nn.Parameter]: A list of trainable parameters in the model.

    Example:
         model = nn.Linear(10, 5)
         trainable_params = get_trainable_params(model)
         len(trainable_params)
        2
    """
    return list(filter(lambda param: param.requires_grad, model.parameters()))


def get_optimizer(args, params: List[nn.Parameter]) -> optim.Optimizer:
    """
    This function returns an optimizer object based on the provided optimization algorithm name.

    Parameters:
        - args: A namespace object containing the following attributes:
            - opt_name: The name of the optimization algorithm.
            - lr: The learning rate for the optimizer.
            - wd: The weight decay for the optimizer.
        - params: A list of parameters for the optimizer.

    Returns:
        - An optimizer object of the specified type.

    Example:
         args = Namespace(opt_name="sgd", lr=0.01, wd=0.0001)
         model = torch.nn.Linear(10, 10)
         params = get_trainable_params(model)
         optimizer = get_optimizer(args, params)
    """
    return create_optimizer_v2(model_or_params=params, opt=args.opt_name, lr=args.lr, weight_decay=args.wd)


def get_lr_scheduler(args, optimizer, num_iters) -> Union[optim.lr_scheduler.SequentialLR, None]:
    """
    This function returns a learning rate scheduler object based on the provided scheduling algorithm name.

    Parameters:
        - args: A namespace object containing the following attributes:
            - sched_name: The name of the scheduling algorithm.
            - warmup_decay: The decay rate for the warmup scheduler.
            - warmup_epochs: The number of epochs for the warmup scheduler.
            - step_size: The step size for the StepLR scheduler.
            - gamma: The gamma for the StepLR scheduler.
            - epochs: The total number of epochs for training.
            - eta_min: The minimum learning rate for the CosineAnnealingLR scheduler.
        - optimizer: The optimizer object to be used with the scheduler.
        - num_iters: The total number of iterations in an epoch

    Returns:
        - A learning rate scheduler object of the specified type, or None if the sched_name is not recognized.

    Example:
         scheduler = get_lr_scheduler(args, optimizer)
         if scheduler is not None:
             for epoch in range(args.epochs):
                scheduler.step()
                train_one_epoch()
                evaluate()
    """
    warmup_lr = optim.lr_scheduler.LinearLR(optimizer, start_factor=args.warmup_decay, total_iters=args.warmup_epochs)
    if args.sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                                                         eta_min=args.eta_min)
    elif args.sched_name == "one_cycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs,
                                                  steps_per_epoch=num_iters)
    else:
        return None

    lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr, scheduler],
                                                   milestones=[args.warmup_epochs])
    return lr_scheduler


# adapted from https://github.com/pytorch/vision/blob/a5035df501747c8fc2cd7f6c1a41c44ce6934db3/references
# /classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """
    Exponential Moving Average (EMA) implementation for model parameters.

    Parameters:
        - model (torch.nn.Module): The model to apply EMA to.
        - decay (float): The decay factor for EMA.
        - device (str, optional): The device to use for EMA. Defaults to "cpu".

    Example:
         model = MyModel()
         ema = ExponentialMovingAverage(model, decay=0.9)
         ema.update_parameters(model)
    """
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def get_logger(logger_name: str, log_file: str, log_level=logging.DEBUG,
               console_level=logging.INFO) -> logging.Logger:
    """
    Creates a logger with a specified name, log file, and log level.
    The logger logs messages to a file and to the console.

    Parameters:
        - logger_name (str): A string representing the name of the logger.
        - log_file (str): A string representing the file path of the log file.
        - log_level (int): The log level for the file handler. Defaults to logging.DEBUG.
        - console_level (int): The log level for the console handler. Defaults to logging.INFO.

    Returns:
        - logging.Logger: A logger object.

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
            - img_src: Path to the source directory.
            - img_dest: Path to the destination directory.
        """
        self.img_src = img_src
        self.img_dest = img_dest

    def get_image_classes(self) -> List[str]:
        """
        Get a list of the classes in the source directory.

        Returns:
            - A list of the classes in the source directory.
        """
        ls = glob(os.path.join(self.img_src, "/*"))

        file_set = []
        pattern = r'[^A-Za-z]+'
        for file in ls:
            file_name = os.path.basename(file).split(".")[0]
            cls_name = re.sub(pattern, '', file_name)
            if cls_name not in file_set:
                file_set.append(cls_name)

        return file_set

    def create_class_dirs(self, class_names: List[str]):
        """
        Create directories for each class in `class_names` under `self.img_dest` directory.

        Parameters:
            - class_names: A list of strings containing the names of the image classes.
        """
        for fdir in class_names:
            os.makedirs(os.path.join(self.img_dest, fdir), exist_ok=True)

    def copy_img_to_dirs(self):
        """
        Copy images from `self.img_src` to corresponding class directories in `self.img_dest`.

        The image file is copied to the class directory whose name is contained in the file name.
        If no class name is found in the file name, the file is not copied.
        """
        class_names = self.get_image_classes()
        for file in glob(os.path.join(self.img_src, "/*")):
            for dir_name in class_names:
                if dir_name in file:
                    shutil.copy(file, os.path.join(self.img_dest, dir_name))
                    break


def create_train_val_test_splits(img_src: str, img_dest: str, ratio: tuple) -> None:
    """
    Split images from `img_src` directory into train, validation, and test sets and save them in `img_dest`
    directory. This will save the images in the appropriate directories based on the train-val-test split ratio.

    Parameters:
        - img_src (str): The source directory containing the images to be split.
        - img_dest (str): The destination directory where the split images will be saved.
        - ratio (tuple): The train, val, test splits. E.g (0.8, 0.1, 0.1)

    Example:
         create_train_val_test_splits("data/images", "data/splits")
    """
    splitfolders.ratio(img_src, output=img_dest, seed=333777999, ratio=ratio)
