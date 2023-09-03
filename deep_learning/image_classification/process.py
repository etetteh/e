import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision

from plotly import express as px
from plotly import graph_objects as go
from sklearn.metrics import auc
from accelerate import Accelerator

import utils


def display_results_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load the results from the results file, convert them to a DataFrame, and sort the DataFrame by a given metric.

    Parameters:
        args:
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
    if args.test_only:
        results_list = utils.read_json_lines_file(os.path.join(args.output_dir, "test_results.jsonl"))
    else:
        results_list = utils.read_json_lines_file(os.path.join(args.output_dir, "results.jsonl"))

    results_df = pd.DataFrame(results_list)

    sorted_results_df = results_df.sort_values(by=[args.sorting_metric], ascending=False)

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


def plot_results(args: argparse.Namespace, model_name: str, classes: List[str], accelerator: Accelerator) -> None:
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
    results_df = display_results_dataframe(args)
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
