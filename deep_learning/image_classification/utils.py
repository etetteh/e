import logging
import os
import random
import re
import shutil
import sys
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import splitfolders
import timm
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.transforms import functional as f


def header(*args):
    msg = " ".join(map(str, args))
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def heading(message):
    header("=" * 99)
    header(message)
    header("=" * 99)


def set_seed_for_all(seed: int) -> None:
    """Sets the seed for NumPy, Python's random module, and PyTorch.

    Parameters:
    - seed (int): The seed to be set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_seed_for_worker(worker_id: Optional[int]) -> Optional[int]:
    """Sets the seed for NumPy and Python's random module for the given worker.

    If no worker ID is provided, uses the initial seed for PyTorch and returns None.

    Parameters:
    - worker_id (Optional[int]): The ID of the worker.

    Returns:
    - Optional[int]: The seed used for the worker, or None if no worker ID was provided.
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


def get_data_augmentation(args) -> Dict[str, Callable]:
    """Returns data augmentation transforms for training and validation sets.

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


def convert_to_channels_first(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor from channels last format to channels first format.
    
    Parameters:
        - image (torch.Tensor): A 4-D or 3-D image tensor.

    Returns:
        - torch.Tensor: The image tensor in channels first format.
    """
    if image.dim() == 4:
        image = image if image.shape[1] == 3 else image.permute(0, 3, 1, 2)
    elif image.dim() == 3:
        image = image if image.shape[0] == 3 else image.permute(2, 0, 1)
    return image


def convert_to_channels_last(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor from channels first format to channels last format.
    
    Parameters:
        - image (torch.Tensor): A 4-D or 3-D image tensor.

    Returns:
        - torch.Tensor: The image tensor in channels last format.
    """
    if image.dim() == 4:
        image = image if image.shape[3] == 3 else image.permute(0, 2, 3, 1)
    elif image.dim() == 3:
        image = image if image.shape[2] == 3 else image.permute(1, 2, 0)
    return image


def get_explain_data_aug() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns the transforms for data augmentation used for explaining the model.

    Returns:
        - A tuple of two transforms representing the data augmentation transforms 
        used for explanation and the inverse of those transforms.
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
    """
    model_names = timm.list_pretrained()

    model_names = [m for m in model_names if str(image_size) in m and model_size in m]

    training_models = [m for m in model_names if
                       isinstance(list(timm.create_model(m).named_modules())[-1][1], torch.nn.Linear)]

    return training_models


def create_linear_head(num_ftrs: int, num_classes: int, dropout: float) -> nn.Sequential:
    """
    Creates a new linear head for the given number of classes and dropout rate.

    Parameters:
    num_ftrs (int): The number of input features for the linear head
    num_classes (int): The number of output classes for the linear head
    dropout (float): The dropout rate

    Returns:
    nn.Sequential: A sequential container with a dropout and linear layer
    """
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_ftrs, num_classes, bias=True)
    )


def get_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    """Returns a pretrained model with a new head and the model name.

    The head of the model is replaced with a new linear layer with the given
    number of classes.

    Parameters:
        dropout (float): The dropout rate
        model_name (str): The name of the model to be created using the `timm` library.
        num_classes (int): The number of classes for the new head of the model.

    Returns:
        Tuple[nn.Module, str]: A tuple containing the modified model and the model name.
    """
    model = timm.create_model(model_name, pretrained=True, scriptable=True, exportable=True)
    freeze_params(model)
    if hasattr(model.head, "in_features"):
        num_ftrs = model.head.in_features
        model.head = create_linear_head(num_ftrs, num_classes, dropout)
        if hasattr(model, "head_dist"):
            model.head_dist = create_linear_head(num_ftrs, num_classes, dropout)
    else:
        num_ftrs = model.head.fc.in_features
        model.head.fc = create_linear_head(num_ftrs, num_classes, dropout)
        if hasattr(model, "head_dist"):
            model.head_dist = create_linear_head(num_ftrs, num_classes, dropout)

    if torch.__version__.startswith("2"):
        model = torch.compile(model.to(memory_format=torch.channels_last))
    else:
        model = model.to(memory_format=torch.channels_last)
    return model


def freeze_params(model: nn.Module) -> None:
    """Freezes the parameters in the given model.

    Parameters:
        model (nn.Module): A PyTorch neural network model.
    """
    for param in model.parameters():
        param.requires_grad = False


def get_class_weights(data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Returns the class weights for the given data loader.

    The class weights are calculated as the inverse frequency of each class in the dataset.

    Parameters:
        data_loader (torch.utils.data.DataLoader): A PyTorch data loader.

    Returns:
        torch.Tensor: A tensor of class weights.
    """
    targets = data_loader.dataset.targets
    class_counts = np.bincount(targets)
    class_weights = len(targets) / (len(data_loader.dataset.classes) * class_counts)
    return torch.Tensor(class_weights)


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Returns a list of trainable parameters in the given model.

    Parameters:
        model (nn.Module): A PyTorch neural network model.

    Returns:
        List[nn.Parameter]: A list of trainable parameters in the model.
    """
    return [param for param in model.parameters() if param.requires_grad]


def get_optimizer(args, params: List[nn.Parameter]) -> Union[optim.SGD, optim.AdamW, None]:
    """
    This function returns an optimizer object based on the provided optimization algorithm name.

    Parameters:
    - args: A namespace object containing the following attributes:
        - opt_name: The name of the optimization algorithm.
        - lr: The learning rate for the optimizer.
        - wd: The weight decay for the optimizer.
    - params: A list of parameters for the optimizer.

    Returns:
    - An optimizer object of the specified type, or None if the opt_name is not recognized.

    Example:
    optimizer = get_optimizer(args, model.parameters())
    """
    if args.opt_name == "sgd":
        return optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)
    elif args.opt_name == "adamw":
        return optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    return None


def get_lr_scheduler(args, optimizer) -> Union[optim.lr_scheduler.LinearLR, optim.lr_scheduler.StepLR,
                                        optim.lr_scheduler.CosineAnnealingLR, optim.lr_scheduler.SequentialLR, None]:
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

    Returns:
    - A learning rate scheduler object of the specified type, or None if the sched_name is not recognized.

    Example:
    scheduler = get_lr_scheduler(args, optimizer)
    """
    warmup_lr = optim.lr_scheduler.LinearLR(optimizer, start_factor=args.warmup_decay, total_iters=args.warmup_epochs)
    if args.sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                                                         eta_min=args.eta_min)
    else:
        return None

    lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr, scheduler],
                                                   milestones=[args.warmup_epochs])
    return lr_scheduler


def get_logger(logger_name: str, log_file: str, log_level=logging.DEBUG,
               console_level=logging.INFO) -> logging.Logger:
    """
    Creates a logger with a specified name, log file, and log level.
    The logger logs messages to a file and to the console.

    Parameters:
    - logger_name : a string representing the name of the logger
    - log_file : a string representing the file path of the log file
    - log_level : representing the log level for the file handler (default = logging.DEBUG)
    - console_level : representing the log level for the console handler (default = logging.INFO)

    Returns:
    - a logger object
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
        """Create directories for each class in `class_names` under `self.img_dest` directory.

        Parameters:
        - class_names: A list of strings containing the names of the image classes.

        Example:
        - create_class_dirs(["dogs", "cats", "birds"])
        This will create the following directories under `self.img_dest`:
        - "dogs"
        - "cats"
        - "birds"
        """
        for fdir in class_names:
            os.makedirs(os.path.join(self.img_dest, fdir), exist_ok=True)

    def copy_img_to_dirs(self):
        """Copy images from `self.img_src` to corresponding class directories in `self.img_dest`.

        The image file is copied to the class directory whose name is contained in the file name.
        If no class name is found in the file name, the file is not copied.
        """
        class_names = self.get_image_classes()
        for file in glob(os.path.join(self.img_src, "/*")):
            for dir_name in class_names:
                if dir_name in file:
                    shutil.copy(file, os.path.join(self.img_dest, dir_name))
                    break


def create_train_val_test_splits(img_src: str, img_dest: str) -> None:
    """Split images from `img_src` directory into train, validation, and test sets and save them in `img_dest`
    directory.

    Parameters:
    - img_src: The source directory containing the images to be split.
    - img_dest: The destination directory where the split images will be saved.

    Example:
    - create_train_val_test_splits("images/source", "images/splits")
    This will create the following directories under `img_dest`:
    - "train"
    - "val"
    - "test"
    And will save the images in the appropriate directories based on the train-val-test split ratio.
    """
    splitfolders.ratio(img_src, output=img_dest, seed=333777999, ratio=(0.8, 0.1, 0.1))
