import argparse
import json
import os
import time
from copy import deepcopy
from typing import Tuple

import torch
from torch.utils import data
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    HammingDistance,
    AUROC,
    Accuracy,
    F1Score,
    Recall,
    Precision,
    ConfusionMatrix,
    ROC,
)
from torchvision import datasets

import utils
import explainability


def train_one_epoch(
        args,
        epoch,
        train_loader,
        model,
        optimizer,
        criterion,
        train_metrics,
        device: torch.device,
) -> Tuple[float, float, float, float, float, float, torch.Tensor]:
    """
    This function trains the model for one epoch and returns the metrics for the epoch.

    Parameters:
    - args: A namespace object containing the following attributes:
        - epochs: The total number of epochs for training.
    - epoch: The current epoch number.
    - train_loader: A DataLoader object for the training dataset.
    - model: The model to be trained.
    - optimizer: The optimizer to be used for training.
    - criterion: The loss function to be used.
    - train_metrics: An object for storing and computing training metrics.
    - device: The device to be used for training.

    Returns:
    - A tuple containing the following training metrics for the epoch:
        - loss: The average loss for the epoch.
        - accuracy: The average accuracy for the epoch.
        - auc: The average AUC for the epoch.
        - f1: The average F1 score for the epoch.
        - recall: The average recall for the epoch.
        - precision: The average precision for the epoch.
        - confusion matrix: The confusion matrix for the epoch.

    Example:
    train_loss, train_acc, train_auc, train_f1, train_recall, train_prec, train_cm = train_one_epoch(
        args, epoch, train_loader, model, optimizer, criterion, train_metrics, device
    )
    """
    model.train()
    for image, target in train_loader:
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.set_grad_enabled(True):
            output = model(image.contiguous(memory_format=torch.channels_last))
            loss = criterion(output, target)
            if len(train_loader.dataset.classes) == 2:
                _, pred = torch.max(output, 1)
            else:
                pred = output
            train_metrics.update(pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    total_train_metrics = train_metrics.compute()
    loss, acc, auc, f1, recall, prec, cm = (
        total_train_metrics["loss"].item(),
        total_train_metrics["acc"].item(),
        total_train_metrics["auc"].item(),
        total_train_metrics["f1"].item(),
        total_train_metrics["recall"].item(),
        total_train_metrics["precision"].item(),
        total_train_metrics["cm"].detach(),
    )

    args.logger.info(
        f"Epoch {epoch + 1}/{args.epochs}: Train Metrics - "
        f"loss: {loss:.4f} | "
        f"accuracy: {acc:.4f} | "
        f"auc: {auc:.4f} | "
        f"f1: {f1:.4f} | "
        f"recall : {recall:.4f} | "
        f"precision : {prec:.4f} | "
        f"Confusion Matrix {cm}"
    )

    train_metrics.reset()
    return loss, acc, auc, f1, recall, prec, cm


def evaluate(
        args,
        epoch,
        val_loader,
        model,
        criterion,
        val_metrics,
        roc_metric,
        device: torch.device,
) -> Tuple[float, float, torch.Tensor, float, float, float, float, torch.Tensor]:
    """
    This function evaluates the model on the validation dataset and returns the metrics.

    Parameters:
    - args: A namespace object containing the following attributes:
        - epochs: The total number of epochs for training.
    - epoch: The current epoch number.
    - val_loader: A DataLoader object for the validation dataset.
    - model: The model to be evaluated.
    - criterion: The loss function to be used.
    - val_metrics: An object for storing and computing validation metrics.
    - roc_metric: An object for computing the ROC curve.
    - device: The device to be used for evaluation.

    Returns:
    - A tuple containing the following validation metrics for the epoch:
        - loss: The average loss for the epoch.
        - accuracy: The average accuracy for the epoch.
        - roc: The ROC curve for the epoch.
        - auc: The average AUC for the epoch.
        - f1: The average F1 score for the epoch.
        - recall: The average recall for the epoch.
        - precision: The average precision for the epoch.
        - confusion matrix: The confusion matrix for the epoch.

    Example:
    val_loss, val_acc, val_roc, val_auc, val_f1, val_recall, val_prec, val_cm = evaluate(
        args, epoch, val_loader, model, criterion, val_metrics, roc_metric, device
    )
    """
    model.eval()
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(image.contiguous(memory_format=torch.channels_last))
            _ = criterion(output, target)

            if len(val_loader.dataset.classes) == 2:
                _, pred = torch.max(output, 1)
                val_metrics.update(pred, target)
                roc_metric.update(output[:, 1], target)
            else:
                val_metrics.update(output, target)
                roc_metric.update(output, target)

        total_val_metrics = val_metrics.compute()
        loss, acc, auc, f1, recall, prec, cm = (
            total_val_metrics["loss"].item(),
            total_val_metrics["acc"].item(),
            total_val_metrics["auc"].item(),
            total_val_metrics["f1"].item(),
            total_val_metrics["recall"].item(),
            total_val_metrics["precision"].item(),
            total_val_metrics["cm"].detach(),
        )
        roc = roc_metric.compute()

        args.logger.info(
            f"Epoch {epoch + 1}/{args.epochs}: Val Metrics - "
            f"loss: {loss:.4f} | "
            f"accuracy: {acc:.4f} | "
            f"auc: {auc:.4f} | "
            f"f1: {f1:.4f} | "
            f"recall: {recall:.4f} | "
            f"precision: {prec:.4f} | "
            f"Confusion Matrix {cm}\n"
        )

    val_metrics.reset()
    roc_metric.reset()
    return loss, acc, roc, auc, f1, recall, prec, cm


def main(args: argparse.Namespace) -> None:
    """Runs the training and evaluation of the model.

    Parameters:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = torch.Generator()
    g.manual_seed(args.seed)
    utils.set_seed_for_all(args.seed)

    data_transforms = utils.get_data_augmentation(args)

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(args.dataset_dir, x), data_transforms[x]
        )
        for x in ["train", "val"]
    }

    samplers = {
        "train": data.RandomSampler(image_datasets["train"]),
        "val": data.SequentialSampler(image_datasets["val"]),
    }

    dataloaders = {
        x: data.DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            sampler=samplers[x],
            num_workers=args.num_workers,
            worker_init_fn=utils.set_seed_for_worker,
            generator=g,
            pin_memory=True,
        )
        for x in ["train", "val"]
    }

    train_loader, val_loader = dataloaders["train"], dataloaders["val"]

    train_weights = utils.get_class_weights(train_loader)

    criterion = torch.nn.CrossEntropyLoss(weight=train_weights, label_smoothing=args.label_smoothing).to(device)
    val_criterion = torch.nn.CrossEntropyLoss().to(device)

    num_classes = len(train_loader.dataset.classes)

    task = "binary" if num_classes == 2 else "multiclass"

    top_k = 1 if task == "multiclass" else None
    average = "macro" if task == "multiclass" else "weighted"

    metric_params = {
        "task": task,
        "average": average,
        "num_classes": num_classes,
        "top_k": top_k,
    }

    metric_collection = MetricCollection({
        "loss": HammingDistance(**metric_params),
        "auc": AUROC(**metric_params),
        "acc": Accuracy(**metric_params),
        "f1": F1Score(**metric_params),
        "recall": Recall(**metric_params),
        "precision": Precision(**metric_params),
        "cm": ConfusionMatrix(**metric_params)
    })

    roc_metric = ROC(**metric_params)

    train_metrics = metric_collection
    val_metrics = metric_collection

    file_path = os.path.join(args.output_dir, "results.jsonl")

    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

    for i, model_name in enumerate(args.models):
        model, name = utils.get_model(args, model_name=model_name, num_classes=num_classes)
        model.to(device)

        params = utils.get_trainable_params(model)
        optimizer = utils.get_optimizer(args, params)
        lr_scheduler = utils.get_lr_scheduler(args, optimizer)

        start_epoch = 0
        best_f1 = 0.0
        best_results = {}

        checkpoint_file = os.path.join(args.output_dir, f"{name}_checkpoint.pth")
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location="cpu")

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            start_epoch = checkpoint["epoch"] + 1
            best_f1 = checkpoint["best_f1"]
            best_results = checkpoint["best_results"]

            if start_epoch == args.epochs:
                args.logger.info("Training completed")
            else:
                args.logger.info(f"Resuming training from epoch {start_epoch}\n")

        start_time = time.time()

        utils.heading(f"Training a {name} model: Model {i + 1} of {len(args.models)}")

        for epoch in range(start_epoch, args.epochs):
            train_one_epoch(args, epoch, train_loader, model, optimizer, criterion, train_metrics, device)

            lr_scheduler.step()

            loss, acc, roc, auc, f1, recall, prec, cm = evaluate(args, epoch, val_loader, model, val_criterion,
                                                                 val_metrics, roc_metric, device)

            if f1 >= best_f1:
                best_f1 = f1
                fpr, tpr, _ = roc
                fpr, tpr = [ff.detach().tolist() for ff in fpr], [tt.detach().tolist() for tt in tpr]

                best_model_state = deepcopy(model.state_dict())
                torch.save(best_model_state, os.path.join(args.output_dir, f"{name}_best_model.pth"))

                best_results = {
                    "model": name,
                    "loss": round(loss, 4),
                    "accuracy": round(acc, 4),
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": round(auc, 4),
                    "f1": round(f1, 4),
                    "recall": round(recall, 4),
                    "precision": round(prec, 4),
                    "confusion matrix": cm.tolist(),
                    "time": f"{(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.0f}s"
                }

            checkpoint = {
                "args": args,
                "epoch": epoch,
                "best_f1": best_f1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "best_results": best_results,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"{name}_checkpoint.pth"))

        elapsed_time = time.time() - start_time

        args.logger.info(f"{name} Training complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
        args.logger.info(f"{name} Best Val F1-score {best_f1:.4f}\n")

        with open(f"{args.output_dir}/results.jsonl", "+a") as file:
            json.dump(best_results, file)
            file.write("\n")

        explainability.process_results(args, model_name)

    args.logger.info(f"All results have been saved at {os.path.abspath(args.output_dir)}")


def get_args():
    """
    Parse and return the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Classification")

    parser.add_argument("--dataset_dir", required=True, type=str, help="Directory of the dataset.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the output files to.")

    parser.add_argument("--model", nargs="*", default=None, help="The name of the model to use")
    parser.add_argument("--model_size", type=str, default="small", help="Size of the model to use",
                        choices=["nano", "tiny", "small", "base", "large"])

    parser.add_argument("--seed", default=999333666, type=int, help="Random seed.")

    parser.add_argument("--crop_size", default=224, type=int, help="Size to crop the input images to.")
    parser.add_argument("--val_resize", default=256, type=int, help="Size to resize the validation images to.")
    parser.add_argument("--mag_bins", default=31, type=int, help="Number of magnitude bins.")
    parser.add_argument("--aug_type", default="rand", type=str, help="Type of data augmentation to use.",
                        choices=["augmix", "rand", "trivial"])
    parser.add_argument("--interpolation", default="bicubic", type=str, help="Type of interpolation to use.",
                        choices=["nearest", "bicubic", "bilinear"])
    parser.add_argument("--hflip", default=0.5, type=float,
                        help="Probability of randomly horizontally flipping the input data.")

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for data loading.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train.")

    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for classifier head")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Amount of label smoothing to use.")

    parser.add_argument("--opt_name", default="adamw", type=str, help="Name of the optimizer to use.",
                        choices=["adamw", "sgd"])
    parser.add_argument("--sched_name", default="cosine", type=str, help="Name of the learning rate scheduler to use.")
    parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--wd", default=1e-4, type=float, help="Weight decay.")
    parser.add_argument("--step_size", default=30, type=int, help="Step size for the learning rate scheduler.")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of epochs for the warmup period.")
    parser.add_argument("--warmup_decay", default=0.1, type=float, help="Decay rate for the warmup learning rate.")
    parser.add_argument("--gamma", default=0.1, type=float, help="Gamma for the learning rate scheduler.")
    parser.add_argument("--eta_min", default=1e-4, type=float,
                        help="Minimum learning rate for the learning rate scheduler.")

    parser.add_argument("--sorting_metric", default="f1", type=str, help="Metric to sort the results by.",
                        choices=["f1", "auc", "accuracy", "precision", "recall"])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory created: {os.path.abspath(args.output_dir)}")
    else:
        print(f"Output directory already exist at: {os.path.abspath(args.output_dir)}")

    if args.model is not None:
        if type(args.model) == list:
            args.models = args.model
        else:
            args.models = [args.model]
    else:
        args.models = sorted(utils.get_model_names(args.crop_size, args.model_size))

    args.logger = utils.get_logger(f"Training and Evaluation of Image Classifiers",
                                   f"{args.output_dir}/training_logs.log")

    main(args)
