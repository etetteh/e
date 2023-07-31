import argparse
import os
import torch
import warnings

import onnxruntime
import torch.nn.functional as f
import torchvision.transforms as transforms

from glob import glob
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import utils


def run_inference(args: argparse.Namespace) -> dict:
    """
    Run inference on a given ONNX model for image classification.

    Parameters:
        args (argparse.Namespace):
            onnx_model_path (str): Path to the ONNX model.
            img_path (str): Path to a single image or a directory containing images to be classified.
            dataset_dir_or_classes_file (str): Path to the directory containing the dataset classes or Path to a text file
                                               containing class names (each class name on a separate line).
            grayscale (bool): Whether to use grayscale images or not
            crop_size (int): The size of the crop for the training and validation sets.
            val_resize (int): The target size for resizing the validation images.
                                 The validation images will be resized to this size while maintaining their aspect ratio.
    Returns:
        dict: A dictionary containing image names as keys and a list of dictionaries with top predicted labels and their
              associated probabilities as values. The list contains at most three dictionaries, showing the top 3
              predicted labels and probabilities if available, otherwise the top 2.

    Example:
        >>> onnx_model_path = "path/to/onnx/model.onnx"
        >>> img_path = "path/to/image.jpg"
        >>> dataset_dir_or_classes_file = "path/to/dataset_classes.txt"
        >>> results = run_inference(onnx_model_path, img_path, dataset_dir_or_classes_file)
        >>> print(results)
        {
            'image.jpg': [
                {'Predicted class': 'cat', 'Probability': 0.68},
                {'Predicted class': 'dog', 'Probability': 0.18},
                {'Predicted class': 'bird', 'Probability': 0.08}
            ]
        }
    """

    if os.path.isfile(args.dataset_dir_or_classes_file):
        with open(args.dataset_dir_or_classes_file, "r") as file_in:
            classes = sorted(file_in.read().splitlines())
    else:
        image_dataset = utils.load_image_dataset(args.dataset_dir_or_classes_file)
        if "label" in image_dataset.column_names["train"]:
            image_dataset = image_dataset.rename_columns({"label": "labels"})
        classes = utils.get_classes(image_dataset["train"])

    data_aug = [
        transforms.Resize(args.val_resize),
        transforms.CenterCrop(args.crop_size),
    ]
    transform_img = transforms.Compose(utils.apply_normalization(args, data_aug))

    if os.path.isdir(args.img_path):
        imgs_path = glob(os.path.join(args.img_path, "*"))
    else:
        imgs_path = [args.img_path]

    imgs = [Image.open(img) for img in imgs_path]
    imgs = [transform_img(img) for img in imgs]
    imgs = [img.unsqueeze_(0) for img in imgs]

    ort_session = onnxruntime.InferenceSession(args.onnx_model_path)

    results = {}
    num_out = range(3) if len(classes) >= 3 else range(2)
    for i, img in enumerate(imgs):
        ort_inputs = {ort_session.get_inputs()[0].name: img.detach().cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        prob = f.softmax(torch.from_numpy(ort_outs[0]), dim=1)
        top_p, top_class = torch.topk(prob, len(classes), dim=1)

        results.update({
            f"{os.path.basename(imgs_path[i])}":
                [
                    {"Predicted class": classes[top_class[0][i].item()], "Probability": round(top_p[0][i].item(), 2)}
                    for i in num_out
                ]
        }
        )

    return results


def get_args():
    parser = argparse.ArgumentParser(description="Run a single inference on a ONNX model for image classification")

    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the output files to.")
    parser.add_argument("--onnx_model_path", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--img_path", type=str, required=True, help="Path to a single image or a directory containing "
                                                                    "images to be classified")
    parser.add_argument("--dataset_dir_or_classes_file", type=str, required=True,
            help="Path to the directory containing the dataset classes or Path to a text file containing class names")
    parser.add_argument('--grayscale', action='store_true', help='Whether to use grayscale images or not')
    parser.add_argument("--crop_size", default=224, type=int, help="Size to crop the input images to.")
    parser.add_argument("--val_resize", default=256, type=int, help="Size to resize the validation images to.")

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    result = run_inference(args)
    utils.write_dictionary_to_json(dictionary=result, file_path=f"{args.output_dir}/inference_results.json")

    print(f"Inference results have been saved to {args.output_dir}/inference_results.json")
