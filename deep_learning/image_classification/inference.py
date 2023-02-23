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


def run_inference(onnx_model_path: str, img_path: str, dataset_dir_or_classes_file: str) -> dict:
    """
    Runs one inference on a given ONNX model and image.

    Parameters:
        - onnx_model_path (str): Path to the ONNX model.
        - img_path (str): Path to a single image or a directory containing images to be classified.
        - dataset_dir_or_classes_file (str): Path to the directory containing the dataset classes or Path to a text file
                                containing class names (each class name on a separate line).

    Returns:
        - dict: A dict of dictionary[ies] containing image name and its predicted label and the associated probability.

    """

    if os.path.isfile(dataset_dir_or_classes_file):
        with open(dataset_dir_or_classes_file, "r") as file_in:
            classes = sorted(file_in.read().splitlines())
    else:
        classes = utils.get_classes(dataset_dir_or_classes_file)

    transform_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ]
       )

    if os.path.isdir(img_path):
        imgs_path = glob(os.path.join(img_path, "*"))
    else:
        imgs_path = [img_path]

    imgs = [Image.open(img) for img in imgs_path]
    imgs = [transform_img(img) for img in imgs]
    imgs = [img.unsqueeze_(0) for img in imgs]

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    results = {}
    for i, img in enumerate(imgs):
        ort_inputs = {ort_session.get_inputs()[0].name: utils.convert_tensor_to_numpy(img)}
        ort_outs = ort_session.run(None, ort_inputs)

        prob = f.softmax(torch.from_numpy(ort_outs[0]), dim=1)
        top_p, top_class = torch.topk(prob, 1, dim=1)

        results.update({
            f"{os.path.basename(imgs_path[i])}": {
                "Predicted Label": classes[top_class.item()],
                "Probability": round(top_p.item() * 100, 2),
            }
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

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    result = run_inference(args.onnx_model_path, args.img_path, args.dataset_dir_or_classes_file)
    utils.write_json_file(dict_obj=result, file_path=f"{args.output_dir}/inference_results.json")

    print(f"Inference results have been saved to {args.output_dir}/inference_results.json")
