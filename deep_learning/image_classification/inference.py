import argparse
import os
import torch
import onnxruntime
import torch.nn.functional as f
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

import utils


def run_one_inference(model_name: str, img_path: str, checkpoint_path: str, dataset_dir: str) -> dict:
    """
    Runs one inference on a given model and image.

    Parameters:
        model_name (str): Name of the model.
        img_path (str): Path to the image.
        checkpoint_path (str): Path to the checkpoint.
        dataset_dir (str): Path to the dataset.

    Returns:
        dict: A dictionary with the predicted label and the probability.

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = utils.get_classes(dataset_dir)

    model = utils.get_model(model_name, len(classes), 0.2)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    filename = os.path.join(os.path.dirname(args.checkpoint_path), f"{model_name}.onnx")

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

    transform_img = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    img = Image.open(img_path)
    img = transform_img(img)
    img.unsqueeze_(0)

    ort_session = onnxruntime.InferenceSession(filename)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)

    prob = f.softmax(torch.from_numpy(ort_outs[0]), dim=1)
    top_p, top_class = prob.topk(1, dim=1)

    return {
        "Predicted Label": classes[top_class.item()],
        "Probability": round(top_p.item() * 100, 2),
    }


def get_args():
    parser = argparse.ArgumentParser(description="Run a single inference on an image classification model")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image to be classified")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    result = run_one_inference(args.model_name, args.img_path, args.checkpoint_path, args.dataset_dir)
    print(result)
