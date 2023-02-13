import argparse
import os
import torch
import warnings

import onnxruntime
import torch.nn.functional as f
from PIL import Image
import torchvision.transforms as transforms

import utils


def run_inference(onnx_model_path: str, imgs_path: str, dataset_dir_or_classes_file: str) -> list:
    """
    Runs one inference on a given ONNX model and image.

    Parameters:
        - onnx_model_path (str): Path to the ONNX model.
        - imgs_paths (str): Paths to the images.
        - dataset_dir_or_classes_file (str): Path to the directory containing the dataset classes or Path to a text file
                                containing class names (each class name on a separate line).

    Returns:
        - dict: A list of dictionary/dictionaries containing the predicted label and the associated probability.

    """

    if os.path.isfile(dataset_dir_or_classes_file):
        with open(dataset_dir_or_classes_file, "r") as file_in:
            classes = sorted(file_in.read().splitlines())
    else:
        classes = utils.get_classes(dataset_dir_or_classes_file)

    transform_img = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

    # if
    imgs = [Image.open(img_path) for img_path in imgs_path]
    imgs = [transform_img(img) for img in imgs]
    imgs = [img.unsqueeze_(0) for img in imgs]

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    results = []
    for i, img in enumerate(imgs):
        ort_inputs = {ort_session.get_inputs()[0].name: utils.convert_tensor_to_numpy(img)}
        ort_outs = ort_session.run(None, ort_inputs)

        prob = f.softmax(torch.from_numpy(ort_outs[0]), dim=1)
        top_p, top_class = torch.topk(prob, 1, dim=1)

        results.append({
            f"image {i}": {
                "Predicted Label": classes[top_class.item()],
                "Probability": round(top_p.item() * 100, 2),
            }
        }
        )

    return results


def get_args():
    parser = argparse.ArgumentParser(description="Run a single inference on a ONNX model for image classification")

    parser.add_argument("--onnx_model_path", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--imgs_paths", nargs="*", required=True, help="Path to the image to be classified")
    parser.add_argument("--dataset_dir_or_classes_file", type=str, required=True,
            help="Path to the directory containing the dataset classes or Path to a text file containing class names")

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    result = run_inference(args.onnx_model_path, args.imgs_paths, args.dataset_dir_or_classes_file)
    print(result)
