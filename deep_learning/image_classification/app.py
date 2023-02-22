import json
import uvicorn
from fastapi import FastAPI, UploadFile
from inference import run_inference

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile):
    """
    Predict the label and confidence of the input image.

    Parameters:
        - file (UploadFile): a JSON file containing the following keys:
            - onnx_model_path (str): Path to ONNX model
            - img_path (str): Path to a single image or a directory containing images to be classified
            - dataset_dir_or_classes_file (str): Path to dataset directory or file with list of classes

    Returns:
        - dict: A dict of dictionary[ies] containing image name and its predicted label and the associated probability.

    """
    input_json = json.loads(await file.read())
    onnx_model_path = input_json['onnx_model_path']
    imgs_path = input_json['img_path']
    dataset_dir_or_classes_file = input_json['dataset_dir_or_classes_file']

    results = run_inference(onnx_model_path, imgs_path, dataset_dir_or_classes_file)

    return results


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
