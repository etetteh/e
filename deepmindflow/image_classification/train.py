import argparse
import json
import os
import shutil
import time
import warnings

import accelerate
from accelerate import (
    Accelerator,
    DeepSpeedPlugin,
    find_executable_batch_size
)
from accelerate.utils import set_seed
from argparse import Namespace
from glob import glob
from typing import Any, Callable, Dict, Optional, Tuple, List

import mlflow
import mlflow.pytorch

import process
import torch
from torch import nn, optim
from torch.utils import data
from torchmetrics import MetricCollection
import torchmetrics.classification as metrics

import utils


def train_one_epoch(
        args: Namespace,
        epoch: int,
        classes: List[str],
        train_loader: data.DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Callable,
        train_metrics: Any,
        accelerator: accelerate.Accelerator,
) -> Dict[str, Any]:
    """
    Trains the model for one epoch and returns the computed metrics.

    Parameters:
        args (Namespace): A namespace object containing training attributes.
        epoch (int): The current epoch number.
        classes (List[str]): The list of class labels.
        train_loader (DataLoader): The training data loader.
        model (nn.Module): The model to be trained.
        optimizer (optim.Optimizer): The optimizer used for training.
        criterion (Callable): The loss function.
        train_metrics (Any): An object for storing and computing training metrics.
        accelerator (accelerate.Accelerator): The accelerator object for distributed training.

    Returns:
        Dict[str, Any]: A dictionary containing the computed metrics for the epoch.
    """
    model.train()
    with accelerator.join_uneven_inputs([model], even_batches=False):
        for idx, batch in enumerate(train_loader):
            images, labels = batch["pixel_values"], batch["labels"]
            images.requires_grad = True

            with accelerator.accumulate([model]):
                optimizer.zero_grad()
                if args.mixup:
                    mixed_images, labels_a, labels_b, lam = utils.apply_mixup(images, labels, alpha=args.mixup_alpha)
                    output = model(mixed_images.contiguous(memory_format=torch.channels_last))
                    loss = lam * criterion(output, labels_a) + (1 - lam) * criterion(output, labels_b)
                elif args.cutmix:
                    mixed_images, labels_a, labels_b, lam = utils.apply_cutmix(images, labels,
                                                                               alpha=args.cutmix_alpha)
                    output = model(mixed_images.contiguous(memory_format=torch.channels_last))
                    loss = lam * criterion(output, labels_a) + (1 - lam) * criterion(output, labels_b)
                else:
                    output = model(images.contiguous(memory_format=torch.channels_last))
                    loss = criterion(output, labels)

                accelerator.backward(loss)

                if args.fgsm:
                    images_grad = images.grad.data
                    adversarial_images = utils.apply_fgsm_attack(images, args.epsilon, images_grad)

                    adversarial_output = model(adversarial_images)
                    adversarial_loss = criterion(adversarial_output, labels)
                    accelerator.backward(adversarial_loss)

                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if len(classes) == 2:
                _, pred = torch.max(output, 1)
            else:
                pred = output
            train_metrics.update(
                accelerator.gather_for_metrics(pred),
                accelerator.gather_for_metrics(labels)
            )

    total_train_metrics = train_metrics.compute()
    loss, acc, auc, f1, recall, prec, cm = (
        total_train_metrics["loss"].item(),
        total_train_metrics["acc"].item(),
        total_train_metrics["auc"].item(),
        total_train_metrics["f1"].item(),
        total_train_metrics["recall"].item(),
        total_train_metrics["precision"].item(),
        total_train_metrics["cm"].detach().cpu().numpy(),
    )

    accelerator.print(
        f"Epoch {epoch + 1}/{args.epochs}: Train Metrics - "
        f"loss: {loss:.4f} | "
        f"accuracy: {acc:.4f} | "
        f"auc: {auc:.4f} | "
        f"f1: {f1:.4f} | "
        f"recall : {recall:.4f} | "
        f"precision : {prec:.4f} | "
        f"Confusion Matrix \n{cm}"
    )

    return total_train_metrics


def evaluate(
        classes: List,
        val_loader: data.DataLoader,
        model: nn.Module,
        val_metrics: Any,
        roc_metric: Any,
        accelerator: accelerate.Accelerator,
) -> Tuple[Dict, torch.Tensor]:
    """
    Evaluates the model on the validation dataset and returns the metrics.

    Parameters:
        classes (List[str]): The list of class labels.
        val_loader (DataLoader): A DataLoader object for the validation dataset.
        model (nn.Module): The model to be evaluated.
        val_metrics (Any): An object for storing and computing validation metrics.
        roc_metric (Any): An object for computing the ROC curve.
        accelerator (accelerate.Accelerator): The accelerator object for distributed training.

    Returns:
        Tuple[Dict[str, Any], torch.Tensor]: A tuple containing a dictionary of computed metrics
                                             and the predicted probabilities.
    """
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch["pixel_values"], batch["labels"]
            output = model(images.contiguous(memory_format=torch.channels_last))

            if len(classes) == 2:
                _, pred = torch.max(output, 1)
                val_metrics.update(accelerator.gather_for_metrics(pred),
                                   accelerator.gather_for_metrics(labels))
                roc_metric.update(accelerator.gather_for_metrics(output[:, 1]),
                                  accelerator.gather_for_metrics(labels))
            else:
                output, labels = accelerator.gather_for_metrics(output), accelerator.gather_for_metrics(labels)
                val_metrics.update(output, labels)
                roc_metric.update(output, labels)

        total_val_metrics = val_metrics.compute()
        loss, acc, auc, f1, recall, prec, cm = (
            total_val_metrics["loss"].item(),
            total_val_metrics["acc"].item(),
            total_val_metrics["auc"].item(),
            total_val_metrics["f1"].item(),
            total_val_metrics["recall"].item(),
            total_val_metrics["precision"].item(),
            total_val_metrics["cm"].detach().cpu().numpy(),
        )
        roc = roc_metric.compute()

        accelerator.print(
            f"Val Metrics - "
            f"loss: {loss:.4f} | "
            f"accuracy: {acc:.4f} | "
            f"auc: {auc:.4f} | "
            f"f1: {f1:.4f} | "
            f"recall: {recall:.4f} | "
            f"precision: {prec:.4f} | "
            f"Confusion Matrix \n{cm}\n"
        )

    return total_val_metrics, roc


def main(args: argparse.Namespace, accelerator) -> None:
    """
    Runs the training and evaluation of the model.

    Parameters:
        args (argparse.Namespace): The command-line and default arguments.
        accelerator: The accelerator object for distributed training.

    Returns:
        None
    """

    # noinspection PyTypeChecker
    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def inner_main_loop(batch_size):
        nonlocal accelerator
        accelerator.free_memory()

        device = accelerator.device
        g = torch.Generator()
        g.manual_seed(args.seed)
        set_seed(args.seed)

        data_transforms = utils.get_data_augmentation(args)
        image_dataset = utils.load_image_dataset(args)
        image_dataset.set_format("torch")

        train_dataset, val_dataset, test_dataset = utils.preprocess_train_eval_data(image_dataset, data_transforms)

        image_datasets = {
            "train": train_dataset,
            "val": val_dataset
        }

        samplers = {
            "train": data.RandomSampler(train_dataset),
            "val": data.SequentialSampler(val_dataset),
        }

        dataloaders = {
            x: data.DataLoader(
                image_datasets[x],
                collate_fn=utils.collate_fn,
                batch_size=batch_size,
                sampler=samplers[x],
                num_workers=args.num_workers,
                worker_init_fn=utils.set_seed_for_worker,
                generator=g,
                pin_memory=True,
            )
            for x in ["train", "val"]
        }

        train_loader, val_loader = dataloaders["train"], dataloaders["val"]
        train_weights = utils.calculate_class_weights(image_dataset)

        test_loader = None
        if args.test_only:
            test_sampler = data.SequentialSampler(test_dataset)

            test_loader = data.DataLoader(
                test_dataset,
                collate_fn=utils.collate_fn,
                batch_size=args.batch_size,
                sampler=test_sampler,
                num_workers=args.num_workers,
                worker_init_fn=utils.set_seed_for_worker,
                generator=g,
                pin_memory=True,
            )

        criterion = torch.nn.CrossEntropyLoss(weight=train_weights, label_smoothing=args.label_smoothing).to(device)

        classes = utils.get_classes(train_dataset)
        num_classes = len(classes)
        task = "binary" if num_classes == 2 else "multiclass"
        top_k = 1 if task == "multiclass" else None
        average = "macro" if task == "multiclass" else "weighted"

        metric_params = {
            "task": task,
            "average": average,
            "num_classes": num_classes,
            "top_k": top_k,
        }

        metric_params_clone = metric_params.copy()
        metric_params_clone.pop("top_k", None)

        metric_collection = MetricCollection({
            "loss": metrics.HammingDistance(**metric_params),
            "auc": metrics.AUROC(**metric_params_clone),
            "acc": metrics.Accuracy(**metric_params),
            "f1": metrics.F1Score(**metric_params),
            "recall": metrics.Recall(**metric_params),
            "precision": metrics.Precision(**metric_params),
            "cm": metrics.ConfusionMatrix(**{"task": task, "num_classes": num_classes})
        })

        roc_metric = metrics.ROC(**{"task": task, "num_classes": num_classes}).to(device)

        train_metrics = metric_collection.to(device)
        val_metrics = metric_collection.to(device)

        run_ids_path = os.path.join(args.output_dir, "run_ids.json")
        if os.path.isfile(run_ids_path):
            run_ids = utils.read_json_file(file_path=run_ids_path)
        else:
            run_ids = None

        file_path = os.path.join(args.output_dir, "results.jsonl")
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

        for i, model_name in enumerate(args.models):
            if args.test_only:
                results_list = utils.read_json_lines_file(os.path.join(args.output_dir, "performance_metrics.jsonl"))
                model_name = results_list[0]["model"]

            if not os.path.isdir(os.path.join(args.output_dir, model_name)):
                os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)

            model = utils.get_pretrained_model(args, model_name=model_name, num_classes=num_classes)
            model = accelerator.prepare_model(model)

            optimizer = utils.get_optimizer(args, model)
            lr_scheduler = utils.get_lr_scheduler(args, optimizer, len(train_loader))

            if args.test_only:
                test_loader = accelerator.prepare_data_loader(test_loader)
            else:
                train_loader, val_loader = accelerator.prepare_data_loader(
                    train_loader), accelerator.prepare_data_loader(val_loader)
                optimizer, lr_scheduler = accelerator.prepare_optimizer(optimizer), accelerator.prepare_scheduler(
                    lr_scheduler)

            start_epoch = 0
            best_f1 = 0.0
            best_results = {}
            best_checkpoints = []

            run_id = utils.get_model_run_id(run_ids, model_name) if run_ids is not None else None

            checkpoint_file = os.path.join(args.output_dir, model_name, "checkpoint.pth")
            best_model_file = os.path.join(args.output_dir, model_name, "best_model")

            mlflow.set_experiment(args.experiment_name)

            with mlflow.start_run(run_id=run_id, run_name=model_name) as run:
                try:
                    mlflow.log_params(vars(args))
                except mlflow.exceptions.MlflowException:
                    pass

                if run_id is None:
                    run_id_pair = {model_name: run.info.run_id}
                    utils.append_dictionary_to_json_file(file_path=run_ids_path, new_dict=run_id_pair)

                if args.test_only:
                    accelerator.print(f"Running evaluation on test data with the best model: {model_name}")
                    with accelerator.main_process_first():
                        if args.avg_ckpts:
                            checkpoint_file = os.path.join(args.output_dir, model_name, "averaged", "best_model.pth")
                            checkpoint = torch.load(checkpoint_file, map_location="cpu")
                        else:
                            checkpoint_file = os.path.join(args.output_dir, model_name, "best_model.pth")
                            checkpoint = torch.load(checkpoint_file, map_location="cpu")
                        model.load_state_dict(checkpoint["model"])
                    total_test_metrics, total_roc_metric = evaluate(classes, test_loader, model, val_metrics,
                                                                    roc_metric, accelerator)

                    test_results = {key: value.detach().tolist() if key == "cm" else round(value.item(), 4) for
                                    key, value in
                                    total_test_metrics.items()}
                    fpr, tpr, _ = total_roc_metric
                    fpr, tpr = [ff.detach().tolist() for ff in fpr], [tt.detach().tolist() for tt in tpr]

                    best_results = {**{"model": model_name, "fpr": fpr, "tpr": tpr}, **test_results}
                    with open(os.path.join(args.output_dir, "test_results.jsonl"), "w") as file:
                        json.dump(best_results, file)
                        file.write("\n")

                    results_df = process.display_results_dataframe(args.output_dir, args.sorting_metric, args.test_only)
                    results_drop = results_df.drop(columns=["loss", "fpr", "tpr", "cm"])
                    results_drop = results_drop.reset_index(drop=True)
                    results_drop.to_json(path_or_buf=os.path.join(args.output_dir, "test_performance_metrics.jsonl"),
                                         orient="records",
                                         lines=True)
                    return

                if os.path.isfile(checkpoint_file):
                    with accelerator.main_process_first():
                        checkpoint = torch.load(checkpoint_file, map_location="cpu")
                        model.load_state_dict(checkpoint["model"])
                        if not args.test_only:
                            optimizer.load_state_dict(checkpoint["optimizer"])
                            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

                        start_epoch = checkpoint["epoch"] + 1
                        best_f1 = checkpoint["best_f1"]
                        best_results = checkpoint["best_results"]

                        if start_epoch == args.epochs:
                            accelerator.print("Training completed")
                        else:
                            accelerator.print(f"Resuming training from epoch {start_epoch}\n")

                start_time = time.time()

                if accelerator.is_main_process:
                    utils.print_header(f"Training a {model_name} model: Model {i + 1} of {len(args.models)}")

                for epoch in range(start_epoch, args.epochs):
                    train_metrics.reset()

                    with accelerator.autocast():
                        total_train_metrics = train_one_epoch(args, epoch, classes, train_loader, model,
                                                              optimizer, criterion, train_metrics, accelerator)

                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()

                    val_metrics.reset()
                    roc_metric.reset()

                    total_val_metrics, total_roc_metric = evaluate(classes, val_loader, model, val_metrics,
                                                                   roc_metric, accelerator)

                    if args.prune:
                        parameters_to_prune = utils.prune_model(model, args.pruning_rate)
                        utils.remove_pruning_reparam(parameters_to_prune)

                    train_results = {key: value.detach().tolist() if key == "cm" else round(value.item(), 4) for
                                     key, value in
                                     total_train_metrics.items()}
                    val_results = {key: value.detach().tolist() if key == "cm" else round(value.item(), 4) for
                                   key, value in
                                   total_val_metrics.items()}

                    accelerator.wait_for_everyone()
                    best_model_list = sorted(glob(os.path.join(args.output_dir, model_name, "best_model_*")))
                    if best_model_list:
                        bottom_k = os.path.splitext(os.path.basename(best_model_list[0]))[0].split("_")[-1]
                        bottom_k = float(bottom_k)

                    if val_results["f1"] >= best_f1:
                        fpr, tpr, _ = total_roc_metric
                        fpr, tpr = [ff.detach().tolist() for ff in fpr], [tt.detach().tolist() for tt in tpr]

                        best_f1 = val_results["f1"]
                        best_results = {**{"model": model_name, "fpr": fpr, "tpr": tpr}, **val_results}

                        accelerator.save({"model": accelerator.get_state_dict(model)},
                                         f"{best_model_file}_{best_f1}.pth")

                        best_checkpoints.append(f"{best_model_file}_{best_f1}.pth")
                    elif val_results["f1"] >= bottom_k:
                        accelerator.save({"model": accelerator.get_state_dict(model)},
                                         f"{best_model_file}_{val_results['f1']}.pth")

                        best_checkpoints.append(f"{best_model_file}_{val_results['f1']}.pth")

                    if len(best_checkpoints) > args.num_ckpts:
                        utils.keep_best_f1_score_files(os.path.join(args.output_dir, model_name), args.num_ckpts)

                    checkpoint = {
                        "args": args,
                        "epoch": epoch,
                        "best_f1": best_f1,
                        "model": accelerator.get_state_dict(model),
                        "optimizer": accelerator.get_state_dict(optimizer),
                        "lr_scheduler": accelerator.get_state_dict(lr_scheduler),
                        "best_results": best_results,
                    }

                    accelerator.save(checkpoint, os.path.join(args.output_dir, model_name, "checkpoint.pth"))

                    if accelerator.is_main_process:
                        mlflow.log_metrics(
                            {f"train_{metric}": value for metric, value in train_results.items() if not metric == "cm"},
                            step=epoch
                        )
                        mlflow.log_metrics(
                            {f"val_{metric}": value for metric, value in val_results.items() if not metric == "cm"},
                            step=epoch
                        )

                elapsed_time = time.time() - start_time
                train_time = f"{elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s"

                accelerator.print(f"{model_name} training completed in {train_time}")
                accelerator.print(f"{model_name} best Val F1-score {best_f1:.4f}\n")

                if args.avg_ckpts:
                    path = os.path.join(args.output_dir, model_name, "averaged")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    avg_model_states = utils.average_checkpoints(glob(f"{best_model_file}*.pth"))
                    torch.save({"model": avg_model_states["model"]}, os.path.join(path, "best_model.pth"))
                else:
                    best_ckpt = sorted(glob(f"{best_model_file}*.pth"))[-1]
                    accelerator.print(f"\nModel to convert as best model: {best_ckpt}\n")
                    shutil.copy(best_ckpt, f"{best_model_file}.pth")

                with open(os.path.join(args.output_dir, "results.jsonl"), "+a") as file:
                    json.dump(best_results, file)
                    file.write("\n")

                process.plot_results(args, model_name, classes, accelerator)

                if args.to_onnx:
                    utils.convert_to_onnx(args, model_name, f"{best_model_file}.pth", num_classes)

                accelerator.free_memory()
                del model, optimizer, lr_scheduler

        results_list = utils.read_json_lines_file(os.path.join(args.output_dir, "performance_metrics.jsonl"))
        best_compare_model_name = results_list[0]["model"]
        best_compare_model_file = os.path.join(args.output_dir,
                                               os.path.join(best_compare_model_name, "averaged") if args.avg_ckpts
                                               else best_compare_model_name,
                                               "best_model.pth")

        utils.convert_to_onnx(args, best_compare_model_name, best_compare_model_file, num_classes)
        accelerator.print(f"Exported best performing model, {best_compare_model_name},"
                          f" to ONNX format. File is located in "
                          f"{os.path.join(args.output_dir, best_compare_model_name)}")

        accelerator.print(f"All results have been saved at {os.path.abspath(args.output_dir)}")

    inner_main_loop()


def get_args():
    """
    Parse and return the command line arguments.

    Returns:
        argparse.Namespace: A namespace containing parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Classification")

    # General Configuration
    parser.add_argument(
        "--experiment_name",
        required=True,
        type=str,
        default="Experiment_1",
        help="The name of the MLflow experiment to organize and categorize experimental runs."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="The path to the dataset directory or the name of a HuggingFace dataset for training and evaluation."
    )
    parser.add_argument(
        "--dataset_kwargs",
        type=str,
        default="",
        help="Path to a JSON file containing keyword arguments (kwargs) specific to a HuggingFace dataset."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The directory where output files, such as trained models and evaluation results, will be saved."
    )

    # Model Configuration
    parser.add_argument(
        "--feat_extract",
        action="store_true",
        help="Enable feature extraction during training when using pretrained models."
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Select a specific model submodule. Choose any of ['beit', 'convnext', 'deit', 'resnet', "
             "'vision_transformer', 'efficientnet', 'xcit', 'regnet', 'nfnet', 'metaformer', 'fastvit', "
             "'efficientvit_msra']"
             "or your favourite from the TIMM  library. Not compatible with --model_size or --model_name."
    )
    parser.add_argument(
        "--model_name",
        nargs="*",
        default=None,
        help="Specify the name(s) of the model(s) from the TIMM library. Not compatible with --model_size or --module."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Specify the model size. Not used when --model_name or --module is specified.",
        choices=["nano", "tiny", "small", "base", "large", "giant"]
    )
    parser.add_argument(
        "--to_onnx",
        action="store_true",
        help="Convert the trained model(s) to ONNX format. If not used, only the best model will be converted."
    )

    # Training Configuration
    # Checkpoint Averaging:
    parser.add_argument(
        "--avg_ckpts",
        action="store_true",
        help="Enable checkpoint averaging during training to stabilize the process."
    )
    parser.add_argument(
        "--num_ckpts",
        type=int,
        default=1,
        help="The number of best checkpoints to save when checkpoint averaging is active."
    )

    # FGSM Adversarial Training
    parser.add_argument(
        "--fgsm",
        action="store_true",
        help="Enable FGSM (Fast Gradient Sign Method) adversarial training to enhance model robustness."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Set the epsilon value for the FGSM attack if FGSM adversarial training is enabled."
    )

    # Pruning
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Enable pruning during training to reduce model complexity and size."
    )
    parser.add_argument(
        "--pruning_rate",
        type=float,
        default=0.25,
        help="Set the pruning rate to control the extent of pruning applied to the model."
    )

    # Random Seed
    parser.add_argument(
        "--seed",
        default=999333666,
        type=int,
        help="Set the random seed for reproducibility of training results."
    )

    # Data Augmentation
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help="Use grayscale images during training."
    )
    parser.add_argument(
        "--crop_size",
        default=224,
        type=int,
        help="The size to which input images will be cropped."
    )
    parser.add_argument(
        "--val_resize",
        default=256,
        type=int,
        help="The size to which validation images will be resized."
    )
    parser.add_argument(
        "--mag_bins",
        default=31,
        type=int,
        help="The number of magnitude bins for augmentation-related operations."
    )
    parser.add_argument(
        "--aug_type",
        default="rand",
        type=str,
        help="The type of data augmentation to use.",
        choices=["augmix", "rand", "trivial"]
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        type=str,
        help="The type of interpolation method to use.",
        choices=["nearest", "bicubic", "bilinear"]
    )
    parser.add_argument(
        "--hflip",
        default=0.5,
        type=float,
        help="The probability of randomly horizontally flipping the input data."
    )

    # Mixup Augmentation
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="Enable mixup augmentation, which enhances training by mixing pairs of samples."
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=1.0,
        help="Set the mixup hyperparameter alpha to control the interpolation factor."
    )

    # Cutmix Augmentation
    parser.add_argument(
        "--cutmix",
        action="store_true",
        help="Enable cutmix augmentation, which combines patches from different images to create new training samples."
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=1.0,
        help="Set the cutmix hyperparameter alpha to control the interpolation factor."
    )

    # Training Parameters
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="The batch size for both training and evaluation."
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="The number of workers for training and evaluation."
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="The number of training epochs, determining how many times the entire dataset will be iterated."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="The dropout rate for the classifier head of the model."
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.1,
        type=float,
        help="The amount of label smoothing to use during training."
    )

    # Optimization and Learning Rate Scheduling
    parser.add_argument(
        "--opt_name",
        default="madgradw",
        type=str,
        help="The optimizer for the training process. Choose any of ['lion', 'madgrad', 'madgradw', 'adamw', "
             "'radabelief', 'adafactor', 'novograd', 'lars', 'lamb', 'rmsprop', 'sgdp'] or any favourite from the "
             "TIMM library"
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="The initial learning rate for the optimizer."
    )
    parser.add_argument(
        "--wd",
        default=1e-4,
        type=float,
        help="The weight decay (L2 regularization) for the optimizer."
    )
    parser.add_argument(
        "--sched_name",
        default="one_cycle",
        type=str,
        help="The learning rate scheduler strategy",
        choices=["step", "cosine", "cosine_wr", "one_cycle"]
    )
    parser.add_argument(
        '--max_lr',
        type=float,
        default=0.1,
        help='The maximum learning rate when using cyclic learning rate scheduling.'
    )
    parser.add_argument(
        "--step_size",
        default=30,
        type=int,
        help="The step size for learning rate adjustments in certain scheduler strategies."
    )
    parser.add_argument(
        "--warmup_epochs",
        default=5,
        type=int,
        help="The number of epochs for the warmup phase of learning rate scheduling."
    )
    parser.add_argument(
        "--warmup_decay",
        default=0.1,
        type=float,
        help="The decay rate for the learning rate during the warmup phase."
    )
    parser.add_argument(
        "--gamma",
        default=0.1,
        type=float,
        help="The gamma parameter used in certain learning rate scheduling strategies."
    )
    parser.add_argument(
        "--eta_min",
        default=1e-5,
        type=float,
        help="The minimum learning rate that the scheduler can reach."
    )
    parser.add_argument(
        "--t0",
        type=int,
        default=5,
        help="The number of iterations for the first restart in learning rate scheduling strategies."
    )

    # Evaluation Metrics and Testing
    parser.add_argument(
        "--sorting_metric",
        default="f1",
        type=str,
        help="The metric by which the model results will be sorted.",
        choices=["f1", "auc", "accuracy", "precision", "recall"]
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Perform testing on the test split only, skipping training when enabled."
    )

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    cfgs = get_args()

    set_seed(cfgs.seed)

    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=2, gradient_clipping=1.0)

    accelerator_var = Accelerator(
        even_batches=True,
        gradient_accumulation_steps=2,
        mixed_precision="fp16",
        deepspeed_plugin=deepspeed_plugin,
    )

    if not os.path.isdir(cfgs.output_dir):
        os.makedirs(cfgs.output_dir, exist_ok=True)
        accelerator_var.print(f"Output directory created: {os.path.abspath(cfgs.output_dir)}")
    else:
        accelerator_var.print(f"Output directory already exists at: {os.path.abspath(cfgs.output_dir)}")

    if cfgs.model_name is not None:
        if isinstance(cfgs.model_name, list):
            cfgs.models = sorted(cfgs.model_name)
        else:
            cfgs.models = [cfgs.model_name]
    else:
        cfgs.models = sorted(utils.get_matching_model_names(cfgs))

    cfgs.lr *= accelerator_var.num_processes
    main(cfgs, accelerator_var)
