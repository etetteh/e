import json
import uvicorn
from argparse import Namespace
from fastapi import FastAPI, UploadFile
from inference import run_inference

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile):
    """
    Predict the label and confidence of the input image.

    Parameters:
        - file (UploadFile): a JSON file

    Returns:
        - dict: A dict of dictionary[ies] containing image name and its predicted label and the associated probability.

    """
    input_json = json.loads(await file.read())
    args = Namespace()
    args.grayscale = input_json["grayscale"]
    args.onnx_model_path = input_json["onnx_model_path"]
    args.img_path = input_json["img_path"]
    args.dataset_dir_or_classes_file = input_json["dataset_dir_or_classes_file"]
    args.crop_size = input_json["crop_size"]
    args.val_resize = input_json["val_resize"]

    results = run_inference(args)

    return results


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
