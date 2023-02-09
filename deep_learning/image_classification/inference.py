import argparse
import torch
import warnings

import onnxruntime
import torch.nn.functional as f
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

import utils


def run_one_inference(onnx_model_path: str, img_path: str, dataset_dir: str) -> dict:
    """
    Runs one inference on a given ONNX model and image.

    Parameters:
        onnx_model_path (str): Path to the ONNX model.
        img_path (str): Path to the image.
        dataset_dir (str): Path to the directory containing the dataset classes.

    Returns:
        dict: A dictionary containing the predicted label and the associated probability.

    """

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a numpy array.

        Parameters:
            tensor (torch.Tensor): The input tensor.

        Returns:
            np.ndarray: The converted numpy array.
        """
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    classes = utils.get_classes(dataset_dir)

    # Preprocess the image
    transform_img = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    img = Image.open(img_path)
    img = transform_img(img)
    img.unsqueeze_(0)

    # Load the ONNX model and run the inference
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Get the prediction and its probability
    prob = f.softmax(torch.from_numpy(ort_outs[0]), dim=1)
    top_p, top_class = torch.topk(prob, 1, dim=1)

    return {
        "Predicted Label": classes[top_class.item()],
        "Probability": round(top_p.item() * 100, 2),
    }


def get_args():
    parser = argparse.ArgumentParser(description="Run a single inference on a ONNX model for image classification")

    parser.add_argument("--onnx_model_path", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image to be classified")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset")

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    result = run_one_inference(args.onnx_model_path, args.img_path, args.dataset_dir)
    print(result)
