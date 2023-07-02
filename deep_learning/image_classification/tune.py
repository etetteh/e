from functools import partial
import argparse
import os
import warnings

import torch
from torch.utils import data
from torchvision import datasets
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

import utils
from train import train_one_epoch, evaluate

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, pb2
from accelerate import Accelerator


def main(config: dict, args: argparse.Namespace) -> None:
    """
    Main function to run the training and validation process.

    Parameters
        - config: Dictionary containing the configuration parameters.
        - args: Argument namespace containing the parsed command line arguments.
    
    Returns
        - None
    """
    if args.tune_aug_type:
        args.aug_type = config["aug_type"]
        
    if args.tune_mag_bins:
        args.mag_bins = config["mag_bins"]
        
    if args.tune_interpolation:
        args.interpolation = config["interpolation"]
        
    if args.tune_dropout:
        args.dropout = config["dropout"]

    if args.tune_ema_steps:
        args.ema_steps = config["ema_steps"]

    if args.tune_mixup:
        args.mixup_alpha = config["mixup_alpha"]

    if args.tune_cutmix:
        args.cutmix_alpha = config["cutmix_alpha"]

    if args.tune_fgsm:
        args.epsilon = config["epsilon"]

    if args.tune_batch_size:
        args.batch_size = config["batch_size"]

    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="fp16")

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

    criterion = torch.nn.CrossEntropyLoss(weight=train_weights,
                                          label_smoothing=config["smoothing"]
                                          if args.tune_smoothing
                                          else args.label_smoothing
                                          ).to(device)

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

    model = utils.get_model(model_name=args.model_name, num_classes=num_classes, dropout=args.dropout)
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

    if args.tune_opt:
        optimizer.param_groups[0]["lr"] = config["lr"]
        optimizer.param_groups[0]["weight_decay"] = config["weight_decay"]
        if args.opt_name == "sgd":
            optimizer.param_groups[0]["momentum"] = config["momentum"]

    if args.tune_sched:
        if args.sched_name == "step":
            lr_scheduler.step_size = config["step_size"]
            lr_scheduler.gamma = config["gamma"]
        elif args.sched_name == "cosine":
            lr_scheduler.T_max = args.epochs - config["warmup_epochs"]
            lr_scheduler.eta_min = config["eta_min"]

    checkpoint_dir = args.output_dir
    checkpoint_file = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    for epoch in range(0, args.epochs):
        train_one_epoch(args, epoch, train_loader, model, model_ema, optimizer, criterion, train_metrics, accelerator)

        if not accelerator.optimizer_step_was_skipped:
            lr_scheduler.step()

        if model_ema:
            evaluate(args, val_loader, model, val_metrics, roc_metric, accelerator, ema=False)
            total_val_metrics, total_roc_metric = evaluate(args, val_loader, model_ema, val_metrics,
                                                           roc_metric, accelerator, ema=True)
        else:
            total_val_metrics, total_roc_metric = evaluate(args, val_loader, model, val_metrics,
                                                           roc_metric, accelerator, ema=False)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "lr_scheduler": lr_scheduler.state_dict()
                 },
                path)

        tune.report(
            auc=total_val_metrics["auc"],
            f1=total_val_metrics["f1"],
            recall=total_val_metrics["recall"],
            prec=total_val_metrics["precision"]
        )

    args.logger.info("Finished Training")


def tune_params(args):
    """
    Tune the hyperparameters of a model based on given arguments.

    Parameters:
        - args (argparse.Namespace): Command-line arguments.

    Returns:
        - tuned results.
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory created: {os.path.abspath(args.output_dir)}")
    else:
        print(f"Output directory already exist at: {os.path.abspath(args.output_dir)}")

    args.logger = utils.get_logger(f"Training and Evaluation of Image Classifiers",
                                   f"{args.output_dir}/training_logs.log")

    config = {}
    hyperparam_mutations = {}

    if args.tune_opt:
        config["lr"] = tune.qloguniform(1e-4, 1e-1, 1e-5)
        config["weight_decay"] = tune.qloguniform(1e-4, 1e-1, 1e-5)

        if args.opt_name == "sgd":
            config["momentum"] = tune.uniform(0.6, 0.99)
        hyperparam_mutations["lr"] = [1e-4, 1e-1]
        hyperparam_mutations["weight_decay"] = [1e-5, 1e-1]

        if args.opt_name == "sgd":
            hyperparam_mutations["momentum"] = [0.001, 1]

    if args.tune_batch_size:
        config["batch_size"] = tune.choice([8, 16, 32, 64])
        hyperparam_mutations["batch_size"] = [8, 64]

    if args.tune_smoothing:
        config["smoothing"] = tune.choice([0.05, 0.1, 0.15])
        hyperparam_mutations["smoothing"] = [0.05, 0.15]

    if args.tune_sched:
        config["warmup_epochs"] = tune.randint(3, 15)
        config["warmup_decay"] = tune.choice([0.1, 0.01, 0.001])

        hyperparam_mutations["warmup_epochs"] = [3, 15]
        hyperparam_mutations["warmup_decay"] = [0.1, 0.001]
        if args.sched_name == "step":
            config["step_size"] = tune.randint(args.epochs // 5, args.epochs // 3)
            config["gamma"] = tune.uniform(0.01, 0.1)
            hyperparam_mutations["step_size"] = [args.epochs // 5, args.epochs // 3]
            hyperparam_mutations["gamma"] = [0.01, 0.1]
        elif args.sched_name == "cosine":
            config["eta_min"] = tune.qloguniform(1e-4, 1e-1, 1e-5)

            hyperparam_mutations["eta_min"] = [1e-4, 1e-1]

    if args.tune_dropout:
        config["dropout"] = tune.choice([0, 0.1, 0.2, 0.3, 0.4])
        hyperparam_mutations["dropout"] = [0, 0.4]

    if args.tune_aug_type:
        config["aug_type"] = tune.choice(["augmix", "rand", "trivial"])

    if args.tune_interpolation:
        config["interpolation"] = tune.choice(["nearest", "bilinear", "bicubic"])

    if args.tune_mag_bins:
        config["mag_bins"] = tune.qrandint(16, 39, 1)
        hyperparam_mutations["mag_bins"] = [16, 39]

    if args.tune_ema_steps:
        config["ema_steps"] = tune.choice([8, 16, 32, 48])
        hyperparam_mutations["ema_steps"] = [8, 48]

    if args.tune_mixup:
        config["mixup_alpha"] = tune.quniform(0.2, 0.6, 0.1)
        hyperparam_mutations["mixup_alpha"] = [0.2, 0.6]

    if args.tune_cutmix:
        config["cutmix_alpha"] = tune.quniform(0.2, 0.6, 0.1)
        hyperparam_mutations["cutmix_alpha"] = [0.2, 0.6]

    if args.tune_fgsm:
        config["epsilon"] = tune.quniform(0.01, 0.1, 0.01)
        hyperparam_mutations["epsilon"] = [0.01, 0.1]

    scheduler = None
    if args.asha:
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=args.epochs * 2,
            grace_period=1,
            reduction_factor=2
        )

    if args.pbt:
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=300.0,
            quantile_fraction=0.35,
            hyperparam_mutations=hyperparam_mutations
        )

    if args.pb2:
        scheduler = pb2.PB2(
            time_attr="training_iteration",
            perturbation_interval=300.0,
            quantile_fraction=0.35,
            hyperparam_bounds=hyperparam_mutations
        )

    reporter = CLIReporter(
        infer_limit=4,
        metric_columns=["auc", "f1", "recall", "prec", "training_iteration"]
    )

    result = tune.run(
        partial(main, args=args),
        name=args.name,
        local_dir=args.output_dir,
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_samples,
        metric="f1",
        mode="max",
        search_alg=args.search_alg if args.asha else None,
        scheduler=scheduler,
        progress_reporter=reporter,
        stop={"f1": 0.9999, "training_iteration": 100},
        reuse_actors=True,
        keep_checkpoints_num=1,
        checkpoint_score_attr="f1",
        resume="AUTO",

    )

    # print(ss)
    best_trial = result.get_best_trial("f1", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation f1: {}".format(
        best_trial.last_result["f1"]))
    print("Best trial final validation auc: {}".format(
        best_trial.last_result["auc"]))

    return result


def get_args():
    """
    Parse and return the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Classification Tuning")

    parser.add_argument("--name", required=True, type=str, help="name of the experiment")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Directory of the dataset.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the output files to.")

    parser.add_argument("--model_name", required=True, type=str, help="The name of the model to use")

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

    parser.add_argument("--mixup", action="store_true", help="Whether to enable mixup or not")
    parser.add_argument("--mixup_alpha", type=float, default=1.0, help="mixup hyperparameter alpha")

    parser.add_argument("--cutmix", action="store_true", help="Whether to enable mixup or not")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="mixup hyperparameter alpha")

    parser.add_argument("--fgsm", action="store_true", help="Whether to enable FGSM adversarial training")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Epsilon value for FGSM attack")

    parser.add_argument("--ema", action="store_true", help="Whether to perform Exponential Moving Average or not")
    parser.add_argument("--ema_steps", type=int, default=32, help="number of iterations to update the EMA model ")
    parser.add_argument("--ema_decay", type=float, default=0.99998, help="EMA decay factor")

    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to search for hyperparameters")
    parser.add_argument("--cpus_per_trial", type=int, default=2, help="Number of CPUs per trial")
    parser.add_argument("--gpus_per_trial", type=int, default=0, help="Number of GPUs per trial")
    parser.add_argument("--search_alg", type=str, default="bohb", help="Hyperparameter search algorithm")

    parser.add_argument("--asha", action="store_true", help="whether to use ASHA optimization algorithm")
    parser.add_argument("--pbt", action="store_true",
                        help="whether to use Population Based Training optimization algorithm")
    parser.add_argument("--pb2", action="store_true",
                        help="whether to use Population Based Training 2 optimization algorithm")

    parser.add_argument("--tune_smoothing", action="store_true", help="whether to tune label smoothing hyperparameter")
    parser.add_argument("--tune_dropout", action="store_true", help="whether to tune dropout rate hyperparameter")
    parser.add_argument("--tune_ema_steps", action="store_true", help="whether to tune ema steps")
    parser.add_argument("--tune_mixup", action="store_true", help="whether to tune mixup")
    parser.add_argument("--tune_cutmix", action="store_true", help="whether to tune cutmix")
    parser.add_argument("--tune_fgsm", action="store_true", help="whether to tune fgsm")
    parser.add_argument("--tune_aug_type", action="store_true", help="whether to tune augmentation type hyperparameter")
    parser.add_argument("--tune_mag_bins", action="store_true", help="whether to tune magnitude bins hyperparameter")
    parser.add_argument("--tune_batch_size", action="store_true", help="whether to tune batch size hyperparameter")
    parser.add_argument("--tune_opt", action="store_true",
                        help="whether to tune optimization algorithm hyperparameters")
    parser.add_argument("--tune_sched", action="store_true",
                        help="whether to tune learning rate schedule hyperparameters")
    parser.add_argument("--tune_interpolation", action="store_true",
                        help="whether to tune image interpolation method hyperparameter")

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    cfgs = get_args()
    tune_params(cfgs)
