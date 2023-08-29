import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from plotly import express as px
from plotly import graph_objects as go
from torch.utils.data import DataLoader
from sklearn.metrics import auc
from accelerate import (
    Accelerator,
    DeepSpeedPlugin,
    FullyShardedDataParallelPlugin,
    find_executable_batch_size
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

import shap
import os

import utils


def display_results_dataframe(output_dir: str, sorting_metric: str) -> pd.DataFrame:
    """
    Load the results from the results file, convert them to a DataFrame, and sort the DataFrame by a given metric.

    Parameters:
        - output_dir: Directory where the results file is stored.
        - sorting_metric: Metric to sort the DataFrame by.

    Returns:
        - A sorted DataFrame of the results.
    Example:
        >>> output_dir = "results"
        >>> sorting_metric = "accuracy"
        >>> results_df = display_results_dataframe(output_dir, sorting_metric)
        # The function will load the results from the "results.jsonl" file in the "results" directory,
        # convert them into a DataFrame, and sort the DataFrame based on the "accuracy" metric in descending order.
        # The sorted DataFrame will be stored in the variable "results_df".
    """
    results_list = utils.read_json_lines_file(os.path.join(output_dir, "results.jsonl"))

    results_df = pd.DataFrame(results_list)

    sorted_results_df = results_df.sort_values(by=[sorting_metric], ascending=False)

    return sorted_results_df


def plot_confusion_matrix(results_df: pd.DataFrame, model_name: str, classes: List[str], output_dir: str) -> None:
    """
    Plot the confusion matrix for a given model.

    Parameters:
        - results_df: DataFrame of results.
        - model_name: Name of the model to plot the confusion matrix for.
        - classes: List of classes in the dataset.
    """
    cm = results_df[results_df["model"] == model_name]["cm"].iloc[0]

    fig = px.imshow(cm,
                    text_auto=True,
                    aspect="auto",
                    x=classes,
                    y=classes,
                    title=f"{model_name} - Confusion Matrix",
                    labels={"x": "Predicted Condition", "y": "Actual Condition", "color": "Score"},
                    )

    # fig.update_xaxes(side="top")
    fig.update_layout(width=750, height=750)
    if output_dir:
        fig.write_html(os.path.join(output_dir, model_name, "confusion_matrix.html"))
        fig.write_image(os.path.join(output_dir, model_name, "confusion_matrix.png"))


def plot_roc_curve(classes: List[str], results_df: pd.DataFrame, model_name: str, output_dir: str) -> None:
    """
    Plots a Receiver Operating Characteristic (ROC) curve using the Plotly library.
    The number of classes determines the format of the plot.
    If there are 2 classes, it plots a single ROC curve with the area under the curve (AUC)
    value displayed in the title.
    If there are more than 2 classes, it plots multiple ROC curves, one for each class,
    with the average AUC value displayed in the title.
    The plot is displayed and can be saved to a html file if the output_dir is provided

    Parameters:
        - classes: a list of strings representing the names of different classes
        - results_df: a DataFrame containing the results of the model(s) being plotted,
            including the false positive rate (fpr) and true positive rate (tpr)
        - model_name: a string representing the name of the model being plotted
        - output_dir: a string representing the directory where the plot will be saved (if provided)

    Returns:
        - None
    Example:
        >>> classes = ["ClassA", "ClassB", "ClassC"]
        >>> results_df = pd.DataFrame({"model": ["MyModel"], "fpr": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
        ...                            "tpr": [[0.6, 0.7, 0.8], [0.7, 0.8, 0.9], [0.8, 0.9, 0.95]]})
        >>> model_name = "MyModel"
        >>> output_dir = "output"
        >>> plot_roc_curve(classes, results_df, model_name, output_dir)
        # The function will plot the ROC curve for each class and save the plot as an HTML file in the 'output' directory.
    """
    num_classes = len(classes)

    if num_classes < 2:
        raise ValueError("Number of classes must be at least 2")

    fig = go.Figure()
    fig.add_shape(
        type='line',
        line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    if num_classes == 2:
        fpr = results_df[results_df["model"] == model_name]["fpr"].iloc[0]
        tpr = results_df[results_df["model"] == model_name]["tpr"].iloc[0]

        auc_val = auc(fpr, tpr)

        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines'))
        title = f"{model_name} - ROC Curve (AUC = {auc_val:.4f})"
    else:
        auc_list = []
        for i in range(len(classes)):
            fpr = results_df[results_df["model"] == model_name]["fpr"].iloc[0][i]
            tpr = results_df[results_df["model"] == model_name]["tpr"].iloc[0][i]

            auc_val = auc(fpr, tpr)

            name = f"{classes[i]} (AUC = {auc_val:.4f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            auc_list.append(auc_val)
        title = f"{model_name} - ROC Curve (Average AUC = {np.mean(auc_list):.4f})"

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=750, height=750
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, model_name, "roc_curve.html"))
        fig.write_image(os.path.join(output_dir, model_name, "roc_curve.png"))


def process_results(args: argparse.Namespace, model_name: str, classes: List[str], accelerator: Accelerator) -> None:
    """
    Processes and saves the performance metrics and plots confusion matrix and ROC curve.

    Parameters:
        args (argparse.Namespace): A Namespace object with the following attributes:
            - output_dir (str): The directory where the results will be saved.
            - sorting_metric (str): The metric to sort the results by.
            - dataset_dir (str): The directory of the dataset.
        model_name (str): The name of the model.
        classes (List[str]): A list of strings representing the classes.
        accelerator (Accelerator): An object representing the accelerator.

    Returns:
        None

    Side Effects:
        - Saves the performance metrics as a JSONL file in the 'args.output_dir'.
        - Prints the model's performance or its performance compared to other models.
        - Plots the confusion matrix and ROC curve and saves them in the 'args.output_dir'.
    Example:
        >>> args = argparse.Namespace(output_dir="results", sorting_metric="accuracy", dataset_dir="data")
        >>> model_name = "MyModel"
        >>> classes = ["class1", "class2", "class3"]
        >>> accelerator = Accelerator()
        >>> process_results(args, model_name, classes, accelerator)
        # The function will process the results, save performance metrics as a JSONL file in the 'results' directory,
        # and print the model's performance or its performance compared to other models. It will also plot the confusion
        # matrix and ROC curve and save them in the 'results' directory.
    """
    results_df = display_results_dataframe(output_dir=args.output_dir, sorting_metric=args.sorting_metric)
    results_drop = results_df.drop(columns=["loss", "fpr", "tpr", "cm"])

    results_drop = results_drop.reset_index(drop=True)

    results_drop.to_json(path_or_buf=os.path.join(args.output_dir, "performance_metrics.jsonl"), orient="records",
                         lines=True)

    if args.model_name and len(args.model_name) == 1:
        accelerator.print(f"\nModel performance:\n{results_drop}\n")
    else:
        accelerator.print(f"\nModel performance against other models:\n{results_drop}\n")

    plot_confusion_matrix(results_df, model_name, classes, args.output_dir)

    plot_roc_curve(classes, results_df, model_name, args.output_dir)


def explain_model(args: argparse.Namespace) -> None:
    """
    Explain the predictions of a given model on a dataset using SHAP values.

    Parameters:
        - args (argparse.Namespace): Arguments passed to the script.
            - dataset_dir (str): Directory of the dataset to use.
            - model_name (str): Name of the model to use.
            - crop_size (int): Size of the random crop applied to the images.
            - batch_size (int): Batch size for data loading.
            - num_workers (int): Number of workers for data loading.
            - n_samples (int): Number of samples to explain.
            - max_evals (int): Maximum number of evaluations for SHAP.
            - topk (int): Number of top-k predictions to plot.
    Returns:
        None

    Side Effects:
        - Plots the SHAP value interpretation for the model predictions on a subset of images from the dataset.

    Example:
        >>> args = argparse.Namespace(
        ...     dataset_dir="data",
        ...     model_name="MyModel",
        ...     crop_size=224,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     n_samples=5,
        ...     max_evals=100,
        ...     topk=5
        ... )
        >>> explain_model(args)
        # The function will use the provided arguments to load the model, dataset, and then explain the model
        # predictions using SHAP values. It will plot the SHAP value interpretation for the top-k predictions on
        # a subset of n_samples images from the dataset. The results will be displayed in the console or as a
        # visualization, depending on the implementation.
    """
    def predict(img: np.ndarray) -> torch.Tensor:
        """
        Predict the class probabilities for an image.

        Parameters:
            - img (np.ndarray): Input image.

        Returns:
            - torch.Tensor: Class probabilities.
        """
        img = utils.to_channels_first(torch.Tensor(img))
        img = img.to(device)
        output = model(img)
        return output

    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=2, gradient_clipping=1.0)
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    )
    accelerator = Accelerator(even_batches=True,
                              gradient_accumulation_steps=2,
                              mixed_precision="fp16",
                              deepspeed_plugin=deepspeed_plugin,
                              fsdp_plugin=fsdp_plugin
                              )

    device = accelerator.device
    transform, inv_transform = utils.get_explanation_transforms()

    image_dataset = utils.load_image_dataset(args)
    classes = utils.get_classes(image_dataset["train"])

    augmentation = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(args.crop_size),
        torchvision.transforms.PILToTensor(),
    ])
    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [
            augmentation(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    val_dataset = image_dataset["test"].with_transform(preprocess_val)

    data_loader = DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             collate_fn=utils.collate_fn
                             )

    batch = next(iter(data_loader))
    images, labels = batch["pixel_values"], batch["labels"]
    images = images.permute(0, 2, 3, 1)
    images = transform(images)

    model = utils.get_pretrained_model(args, model_name=args.model_name, num_classes=len(classes))

    checkpoint_file = os.path.join(os.path.join(args.output_dir, "best_model.pth"))
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    if "n_averaged" in checkpoint["model"]:
        del checkpoint["model"]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["model"], "module.")
    model.load_state_dict(checkpoint["model"])

    model.eval()
    model.to(device)

    masker_blur = shap.maskers.Image(f"blur{args.crop_size // 2, args.crop_size // 2}", images[0].shape)
    explainer = shap.Explainer(predict, masker_blur, output_names=classes)

    shap_values = explainer(images[:args.n_samples],
                            max_evals=args.max_evals,
                            batch_size=args.batch_size,
                            outputs=shap.Explanation.argsort.flip[:args.topk]
                            )

    if args.n_samples == 1:
        shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
        shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]
    else:
        shap_values.data = inv_transform(shap_values.data).cpu().numpy()
        shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    shap.image_plot(shap_values=shap_values.values,
                    pixel_values=shap_values.data,
                    labels=shap_values.output_names,
                    true_labels=[classes[idx] for idx in labels[:len(labels)]],
                    aspect=0.15,
                    hspace=0.15,
                    show=False,
                    )

    filename = f"{args.output_dir}/shap_explanation.png"

    # Save the SHAP explanation plot as a PNG image
    plt.savefig(filename, bbox_inches='tight', format='png')
    plt.close(fig)

    accelerator.print(f"Saved SHAP explanation plot as {filename}")
