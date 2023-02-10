import json
import os
import uvicorn

import numpy as np
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms
from PIL import Image

import onnxruntime
import utils

from fastapi import FastAPI, UploadFile


app = FastAPI()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy ndarray
    Args:
    tensor (torch.Tensor): torch tensor to be converted

    Returns:
    np.ndarray: numpy ndarray converted from tensor
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


@app.post("/predict")
async def predict(file: UploadFile):
    """
    Predict the label and confidence of the input image.

    Parameters:
     - file (UploadFile): a JSON file containing the following keys:
     - onnx_model_path (str): Path to ONNX model
     - img_path (str): Path to input image
     - dataset_dir_or_classes_file (str): Path to dataset directory or file with list of classes

    Returns:
    dict: A dictionary containing the following keys:
        "Predicted Label": str, the predicted label of the image
        "Probability": float, the confidence of the prediction
    """
    input_json = json.loads(await file.read())
    onnx_model_path = input_json['onnx_model_path']
    img_path = input_json['img_paths']
    dataset_dir_or_classes_file = input_json['dataset_dir_or_classes_file']

    if os.path.isfile(dataset_dir_or_classes_file):
        with open(dataset_dir_or_classes_file, "r") as file_in:
            classes = sorted(file_in.read().splitlines())
    else:
        classes = utils.get_classes(dataset_dir_or_classes_file)

    transform_img = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

    imgs = [Image.open(img_path) for img_path in img_path]
    imgs = [transform_img(img) for img in imgs]
    imgs = [img.unsqueeze_(0) for img in imgs]

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    results = []
    for i, img in enumerate(imgs):
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
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


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
