import argparse
import json
import os
import shutil
import time
import warnings

from copy import deepcopy
from glob import glob
from typing import Tuple, Dict

import mlflow
import mlflow.pytorch

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
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from accelerate import Accelerator, FullyShardedDataParallelPlugin, find_executable_batch_size

import utils
import explainability


def train_one_epoch(
        args,
        epoch,
        train_loader,
        model,
        model_ema,
        optimizer,
        criterion,
        train_metrics,
        accelerator
) -> Dict:
    """
    This function trains the model for one epoch and returns the metrics for the epoch.

    Parameters:
        - args: A namespace object containing the following attributes:
            - epochs: The total number of epochs for training.
        - epoch: The current epoch number.
        - train_loader: A DataLoader object for the training dataset.
        - model: The model to be trained.
        - model_ema: The exponential moving average model for model updates.
        - optimizer: The optimizer to be used for training.
        - criterion: The loss function to be used.
        - train_metrics: An object for storing and computing training metrics.
        - accelerator: The accelerator object for distributed training.

    Returns:
        Dict: A dictionary containing the metrics computed during training.
    """
    model.train()
    for idx, (images, targets) in enumerate(train_loader):
        images.requires_grad = True

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            with accelerator.autocast():
                if args.mixup:
                    mixed_images, targets_a, targets_b, lam = utils.mixup_data(images, targets, alpha=args.mixup_alpha)
                    output = model(mixed_images.contiguous(memory_format=torch.channels_last))
                    loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)
                elif args.cutmix:
                    mixed_images, targets_a, targets_b, lam = utils.cutmix_data(images, targets,
                                                                                alpha=args.cutmix_alpha)
                    output = model(mixed_images.contiguous(memory_format=torch.channels_last))
                    loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)
                else:
                    output = model(images.contiguous(memory_format=torch.channels_last))
                    loss = criterion(output, targets)

            if args.fgsm:
                image_grad = images.grad.data
                images_adversarial = utils.fgsm_attack(images, args.epsilon, image_grad)

                with accelerator.autocast():
                    output_adversarial = model(images_adversarial)
                    loss_adversarial = criterion(output_adversarial, targets)
                    loss = loss + loss_adversarial

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        if model_ema and idx % args.ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.warmup_epochs:
                model_ema.n_averaged.fill_(0)

        if len(train_loader.dataset.classes) == 2:
            _, pred = torch.max(output, 1)
        else:
            pred = output
        train_metrics.update(accelerator.gather_for_metrics(pred),
                             accelerator.gather_for_metrics(targets)
                             )

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

    return total_train_metrics


def evaluate(
        args,
        val_loader,
        model,
        val_metrics,
        roc_metric,
        accelerator,
        ema,
) -> Tuple[Dict, torch.Tensor]:
    """
    This function evaluates the model on the validation dataset and returns the metrics.

    Parameters:
        - args: A namespace object containing the following attributes:
        - val_loader: A DataLoader object for the validation dataset.
        - model: The model to be evaluated.
        - val_metrics: An object for storing and computing validation metrics.
        - roc_metric: An object for computing the ROC curve.
        - device: The device to be used for evaluation.
        - ema: Whether evaluation is being performed on model_ema or not
        - accelerator: The accelerator object for distributed training.

    Returns:
       - Tuple: A tuple containing a dictionary of computed metrics and the predicted probabilities.
    """
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            output = model(images.contiguous(memory_format=torch.channels_last))

            if len(val_loader.dataset.classes) == 2:
                _, pred = torch.max(output, 1)
                val_metrics.update(accelerator.gather_for_metrics(pred),
                                   accelerator.gather_for_metrics(targets))
                roc_metric.update(accelerator.gather_for_metrics(output[:, 1]),
                                  accelerator.gather_for_metrics(targets))
            else:
                output, targets = accelerator.gather_for_metrics(output), accelerator.gather_for_metrics(targets)
                val_metrics.update(output, targets)
                roc_metric.update(output, targets)

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
            f"{'EMA ' if ema else ' '}Val Metrics - "
            f"loss: {loss:.4f} | "
            f"accuracy: {acc:.4f} | "
            f"auc: {auc:.4f} | "
            f"f1: {f1:.4f} | "
            f"recall: {recall:.4f} | "
            f"precision: {prec:.4f} | "
            f"Confusion Matrix {cm}\n"
        )

    return total_val_metrics, roc


def main(args: argparse.Namespace) -> None:
    """
    Runs the training and evaluation of the model.

    Parameters:
        - args (argparse.Namespace): The command-line and default arguments.

    Returns:
        - None
    """
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    )
    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="fp16", fsdp_plugin=fsdp_plugin)

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def inner_main_loop(batch_size):
        nonlocal accelerator
        accelerator.free_memory()

        device = accelerator.device
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
        train_weights = utils.get_class_weights(train_loader)

        test_loader = None
        if os.path.isdir(os.path.join(args.dataset_dir, "test")) and args.test_only:
            test_dataset = datasets.ImageFolder(
                os.path.join(args.dataset_dir, "test"),
                data_transforms["val"]
            )

            test_loader = data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                sampler=samplers["val"],
                num_workers=args.num_workers,
                worker_init_fn=utils.set_seed_for_worker,
                generator=g,
                pin_memory=True,
            )

        criterion = torch.nn.CrossEntropyLoss(weight=train_weights, label_smoothing=args.label_smoothing).to(device)

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

        roc_metric = ROC(**metric_params).to(device)

        train_metrics = metric_collection.to(device)
        val_metrics = metric_collection.to(device)

        run_ids_path = os.path.join(args.output_dir, "run_ids.json")
        if os.path.isfile(run_ids_path):
            run_ids = utils.load_json_file(file_path=run_ids_path)
        else:
            run_ids = None

        file_path = os.path.join(args.output_dir, "results.jsonl")
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

        for i, model_name in enumerate(args.models):
            if not os.path.isdir(os.path.join(args.output_dir, model_name)):
                os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)

            model = utils.get_model(model_name=model_name, num_classes=num_classes, dropout=args.dropout)
            model = accelerator.prepare_model(model)

            params = utils.get_trainable_params(model)
            optimizer = utils.get_optimizer(args, params)
            lr_scheduler = utils.get_lr_scheduler(args, optimizer, len(train_loader))

            optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(optimizer, train_loader, val_loader,
                                                                                    lr_scheduler)

            model_ema = None
            if args.ema:
                adjust = args.batch_size * args.ema_steps / args.epochs
                alpha = 1.0 - args.ema_decay
                alpha = min(1.0, alpha * adjust)
                model_ema = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

            start_epoch = 0
            best_f1 = 0.0
            best_results = {}
            best_checkpoints = []

            run_id = utils.get_run_id(run_ids, model_name) if run_ids is not None else None

            checkpoint_file = os.path.join(args.output_dir, model_name, "checkpoint.pth")
            best_model_file = os.path.join(args.output_dir, model_name, "best_model")

            mlflow.set_experiment(args.experiment_name)

            with mlflow.start_run(run_id=run_id, run_name=model_name) as run:
                try:
                    mlflow.log_params(vars(args))
                except mlflow.exceptions.MlflowException:
                    pass
                mlflow.pytorch.log_model(model, model_name)

                if run_id is None:
                    run_id_pair = {model_name: run.info.run_id}
                    utils.append_dict_to_json_file(file_path=run_ids_path, new_dict=run_id_pair)

                if os.path.isfile(checkpoint_file):
                    checkpoint = torch.load(checkpoint_file, map_location="cpu")
                    model.load_state_dict(checkpoint["model"])
                    if not args.test_only:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                    if model_ema:
                        model_ema.load_state_dict(checkpoint["model_ema"])

                    start_epoch = checkpoint["epoch"] + 1
                    best_f1 = checkpoint["best_f1"]
                    best_results = checkpoint["best_results"]

                    if start_epoch == args.epochs:
                        args.logger.info("Training completed")
                    else:
                        args.logger.info(f"Resuming training from epoch {start_epoch}\n")

                if args.test_only:
                    if args.avg_ckpts:
                        checkpoint_file = os.path.join(args.output_dir, model_name, "averaged", "best_model.pth")
                        checkpoint = torch.load(checkpoint_file, map_location="cpu")
                    else:
                        checkpoint_file = os.path.join(args.output_dir, model_name, "best_model.pth")
                        checkpoint = torch.load(checkpoint_file, map_location="cpu")
                    model.load_state_dict(checkpoint["model"])
                    if model_ema:
                        evaluate(args, test_loader, model_ema, val_metrics, roc_metric, accelerator, ema=True)
                    else:
                        evaluate(args, test_loader, model, val_metrics, roc_metric, accelerator, ema=False)
                    return

                start_time = time.time()

                utils.heading(f"Training a {model_name} model: Model {i + 1} of {len(args.models)}")

                for epoch in range(start_epoch, args.epochs):
                    train_metrics.reset()

                    try:
                        total_train_metrics = train_one_epoch(args, epoch, train_loader, model, model_ema, optimizer,
                                                              criterion, train_metrics, accelerator)
                    except ValueError:
                        continue

                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()

                    val_metrics.reset()
                    roc_metric.reset()

                    if model_ema:
                        evaluate(args, val_loader, model, val_metrics, roc_metric, accelerator, ema=False)
                        total_val_metrics, total_roc_metric = evaluate(args, val_loader, model_ema, val_metrics,
                                                                       roc_metric, accelerator, ema=True)
                    else:
                        total_val_metrics, total_roc_metric = evaluate(args, val_loader, model, val_metrics,
                                                                       roc_metric, accelerator, ema=False)

                    if args.prune:
                        parameters_to_prune = utils.prune_model(model, args.pruning_rate)
                        utils.remove_pruning_reparam(parameters_to_prune)

                    train_results = {key: val.detach().tolist() if key == "cm" else round(val.item(), 4) for key, val in
                                     total_train_metrics.items()}
                    val_results = {key: val.detach().tolist() if key == "cm" else round(val.item(), 4) for key, val in
                                   total_val_metrics.items()}

                    if val_results["f1"] >= best_f1:
                        fpr, tpr, _ = total_roc_metric
                        fpr, tpr = [ff.detach().tolist() for ff in fpr], [tt.detach().tolist() for tt in tpr]

                        best_f1 = val_results["f1"]
                        best_results = {**{"model": model_name, "fpr": fpr, "tpr": tpr}, **val_results}

                        if model_ema:
                            best_model_state = deepcopy(model_ema.state_dict())
                        else:
                            best_model_state = deepcopy(model.state_dict())
                        accelerator.save({"model": best_model_state}, f"{best_model_file}_{best_f1}.pth")

                        best_checkpoints.append(f"{best_model_file}_{best_f1}.pth")

                    checkpoint = {
                        "args": args,
                        "epoch": epoch,
                        "best_f1": best_f1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "best_results": best_results,
                    }
                    if model_ema:
                        checkpoint["model_ema"] = model_ema.state_dict()

                    accelerator.save(checkpoint, os.path.join(args.output_dir, model_name, "checkpoint.pth"))

                    if len(best_checkpoints) > args.num_ckpts:
                        checkpoint_path_to_del = best_checkpoints.pop(0)
                        if os.path.exists(checkpoint_path_to_del):
                            os.remove(checkpoint_path_to_del)

                    mlflow.log_metrics(
                        {f"train_{metric}": value for metric, value in train_results.items() if not metric == "cm"},
                        step=epoch)
                    mlflow.log_metrics(
                        {f"val_{metric}": value for metric, value in val_results.items() if not metric == "cm"},
                        step=epoch)

            elapsed_time = time.time() - start_time
            train_time = f"{elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s"

            args.logger.info(f"{model_name} training completed in {train_time}")
            args.logger.info(f"{model_name} best Val F1-score {best_f1:.4f}\n")

            if args.avg_ckpts:
                path = os.path.join(args.output_dir, model_name, "averaged")
                if not os.path.exists(path):
                    os.makedirs(path)
                avg_model_states = utils.average_checkpoints(glob(f"{best_model_file}*.pth"))
                torch.save({"model": avg_model_states["model"]}, os.path.join(path, "best_model.pth"))
            else:
                best_ckpt = sorted(glob(f"{best_model_file}*.pth"))[-1]
                print(f"\nModel to convert as best model: {best_ckpt}\n")
                shutil.copy(best_ckpt, f"{best_model_file}.pth")

            with open(os.path.join(args.output_dir, "results.jsonl"), "+a") as file:
                json.dump(best_results, file)
                file.write("\n")

            explainability.process_results(args, model_name)

            accelerator.free_memory()
            del model, optimizer, lr_scheduler

        results_list = utils.load_json_lines_file(os.path.join(args.output_dir, "performance_metrics.jsonl"))
        best_compare_model_name = results_list[0]["model"]
        best_compare_model_file = os.path.join(args.output_dir,
                                               os.path.join(best_compare_model_name, "averaged") if args.avg_ckpts
                                               else best_compare_model_name,
                                               "best_model.pth")

        utils.convert_to_onnx(best_compare_model_name, best_compare_model_file, num_classes, args.dropout,
                              args.crop_size)
        args.logger.info(f"Exported best performing model, {best_compare_model_name},"
                         f" to ONNX format. File is located in "
                         f"{os.path.join(args.output_dir, best_compare_model_name)}")

        args.logger.info(f"All results have been saved at {os.path.abspath(args.output_dir)}")

    inner_main_loop()


def get_args():
    """
    Parse and return the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Classification")

    parser.add_argument("--experiment_name", required=True, type=str, default="Experiment_1",
                        help="Name of the MLflow experiment")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Directory of the dataset.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the output files to.")

    parser.add_argument("--model_name", nargs="*", default=None, help="The name of the model to use")
    parser.add_argument("--model_size", type=str, default="small", help="Size of the model to use",
                        choices=["nano", "tiny", "small", "base", "large", "giant"])

    parser.add_argument("--avg_ckpts", action="store_true", help="Whether to enable checkpoint averaging or not.")
    parser.add_argument("--num_ckpts", type=int, default=5, help="Number of best checkpoints to save")

    parser.add_argument("--mixup", action="store_true", help="Whether to enable mixup or not")
    parser.add_argument("--mixup_alpha", type=float, default=1.0, help="mixup hyperparameter alpha")

    parser.add_argument("--cutmix", action="store_true", help="Whether to enable mixup or not")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="mixup hyperparameter alpha")

    parser.add_argument("--fgsm", action="store_true", help="Whether to enable FGSM adversarial training")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Epsilon value for FGSM attack")

    parser.add_argument("--ema", action="store_true", help="Whether to perform Exponential Moving Average or not")
    parser.add_argument("--ema_steps", type=int, default=32, help="number of iterations to update the EMA model ")
    parser.add_argument("--ema_decay", type=float, default=0.99998, help="EMA decay factor")

    parser.add_argument("--prune", action="store_true", help="Whether to perform pruning or not")
    parser.add_argument("--pruning_rate", type=float, default=0.25, help="Pruning rate")

    parser.add_argument("--seed", default=999333666, type=int, help="Random seed.")

    parser.add_argument("--crop_size", default=224, type=int, help="Size to crop the input images to.")
    parser.add_argument("--val_resize", default=256, type=int, help="Size to resize the validation images to.")
    parser.add_argument("--mag_bins", default=31, type=int, help="Number of magnitude bins.")
    parser.add_argument("--aug_type", default="rand", type=str, help="Type of data augmentation to use.",
                        choices=["augmix", "rand", "trivial"])
    parser.add_argument("--interpolation", default="bilinear", type=str, help="Type of interpolation to use.",
                        choices=["nearest", "bicubic", "bilinear"])
    parser.add_argument("--hflip", default=0.5, type=float,
                        help="Probability of randomly horizontally flipping the input data.")

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for data loading.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train.")

    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for classifier head")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Amount of label smoothing to use.")

    parser.add_argument("--opt_name", default="madgrad", type=str, help="Name of the optimizer to use.",)
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--wd", default=1e-4, type=float, help="Weight decay.")

    parser.add_argument("--sched_name", default="one_cycle", type=str, help="Name of the learning rate scheduler "
                        "to use.", choices=["step", "cosine", "one_cycle"])
    parser.add_argument('--max_lr', type=float, default=0.1, help='Maximum learning rate')
    parser.add_argument("--step_size", default=30, type=int, help="Step size for the learning rate scheduler.")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of epochs for the warmup period.")
    parser.add_argument("--warmup_decay", default=0.1, type=float, help="Decay rate for the warmup learning rate.")
    parser.add_argument("--gamma", default=0.1, type=float, help="Gamma for the learning rate scheduler.")
    parser.add_argument("--eta_min", default=1e-4, type=float,
                        help="Minimum learning rate for the learning rate scheduler.")

    parser.add_argument("--sorting_metric", default="f1", type=str, help="Metric to sort the results by.",
                        choices=["f1", "auc", "accuracy", "precision", "recall"])

    parser.add_argument("--test_only", action="store_true", help="Whether to enable testing the trained model or not.")

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    cfgs = get_args()

    if not os.path.isdir(cfgs.output_dir):
        os.makedirs(cfgs.output_dir, exist_ok=True)
        print(f"Output directory created: {os.path.abspath(cfgs.output_dir)}")
    else:
        print(f"Output directory already exists at: {os.path.abspath(cfgs.output_dir)}")

    if cfgs.model_name is not None:
        if type(cfgs.model_name) == list:
            cfgs.models = cfgs.model_name
        else:
            cfgs.models = [cfgs.model_name]
    else:
        cfgs.models = sorted(utils.get_model_names(cfgs.crop_size, cfgs.model_size))

    cfgs.logger = utils.get_logger(f"Training and Evaluation of Image Classifiers",
                                   os.path.join(cfgs.output_dir, "training_logs.log")
                                   )

    main(cfgs)
