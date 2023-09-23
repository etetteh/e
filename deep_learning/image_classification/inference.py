import argparse
import os
import torch
import warnings

import onnxruntime
import torch.nn.functional as f
import torchvision.transforms as transforms

from glob import glob
from PIL import Image

import utils


def run_inference(args: argparse.Namespace) -> dict:
    """
    Run inference on a given ONNX model for image classification.

    Args:
        args (argparse.Namespace):
            onnx_model_path (str): Path to the ONNX model.
            img_path (str): Path to a single image or a directory containing images to be classified.
            dataset_dir_or_classes_file (str): Path to the directory containing the dataset classes or Path to a text file
                                               containing class names (each class name on a separate line).
            grayscale (bool): Whether to use grayscale images or not.
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
        args.dataset = args.dataset_dir_or_classes_file
        image_dataset = utils.load_image_dataset(args)
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
    """
    Parse command-line arguments for running inference with an ONNX model.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.

    Example:
        >>> args = get_args()
        >>> print(args.img_path)
    """
    parser = argparse.ArgumentParser(description="Run inference with an ONNX model")

    # Required arguments
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save the inference results."
    )
    parser.add_argument(
        "--onnx_model_path",
        required=True,
        type=str,
        help="Path to the ONNX model file for inference."
    )
    parser.add_argument(
        "--img_path",
        required=True,
        type=str,
        help="Path to input image(s) or a directory containing images to classify."
    )
    parser.add_argument(
        "--dataset_dir_or_classes_file",
        required=True,
        type=str,
        help="Path to the directory containing dataset classes or a text file with class names."
    )

    # Optional arguments
    parser.add_argument(
        "--dataset_kwargs",
        type=str,
        default="",
        help="Path to a JSON file with dataset-specific keyword arguments."
    )
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Use grayscale images during inference.'
    )
    parser.add_argument(
        "--crop_size",
        default=224,
        type=int,
        help="Size to crop input images during inference."
    )
    parser.add_argument(
        "--val_resize",
        default=256,
        type=int,
        help="Size to which validation images will be resized during inference."
    )

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    result = run_inference(args)
    utils.write_dictionary_to_json(dictionary=result, file_path=f"{args.output_dir}/inference_results.json")

    print(f"Inference results have been saved to {args.output_dir}/inference_results.json")
