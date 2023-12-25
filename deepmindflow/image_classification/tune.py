import argparse
import os
from functools import partial
import warnings

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

import utils
from train import train_one_epoch, evaluate

from accelerate import (
    Accelerator,
    find_executable_batch_size,
    FullyShardedDataParallelPlugin
)
from accelerate.utils import set_seed
# noinspection PyProtectedMember
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

import tempfile
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2

from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.flaml import CFO


def tune_classifier(config, args):
    if torch.cuda.is_available():
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
    else:
        fsdp_plugin = None

    accelerator = Accelerator(
        even_batches=True,
        mixed_precision=args.mixed_precision,
        fsdp_plugin=fsdp_plugin
    )

    if args.tune_batch_size:
        args.batch_size = config["batch_size"]

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def inner_main_loop(batch_size):
        if args.tune_opt:
            args.lr = config["lr"]
            args.wd = config["weight_decay"]

        if args.tune_smoothing:
            args.label_smoothing = config["smoothing"]

        if args.tune_sched:
            args.warmup_epochs = config["warmup_epochs"]
            args.warmup_decay = config["warmup_decay"]

            if args.sched_name == "step":
                args.step_size = config["step_size"]
                args.gamma = config["gamma"]
            elif args.sched_name == "cosine":
                args.eta_min = config["eta_min"]
            elif args.sched_name == "cosine_wr":
                args.t0 = config["t0"]
                args.eta_min = config["eta_min"]
            elif args.sched_name == "one_cycle":
                args.max_lr = config["max_lr"]

        if args.tune_dropout:
            args.dropout = config["dropout"]

        if args.tune_aug_type:
            args.aug_type = config["aug_type"]

        if args.tune_interpolation:
            args.interpolation = config["interpolation"]

        if args.tune_mag_bins:
            args.mag_bins = config["mag_bins"]

        if args.tune_mixup:
            args.mixup_alpha = config["mixup_alpha"]

        if args.tune_cutmix:
            args.cutmix_alpha = config["cutmix_alpha"]

        if args.tune_fgsm:
            args.epsilon = config["epsilon"]

        if args.tune_prune:
            args.pruning_rate = config["pruning_rate"]

        nonlocal accelerator
        gradient_accumulation_steps = args.batch_size // batch_size
        accelerator.gradient_accumulation_steps = gradient_accumulation_steps

        accelerator.free_memory()

        set_seed(args.seed)
        device = accelerator.device
        g = torch.Generator()
        g.manual_seed(args.seed)

        image_dataset, train_dataset, val_dataset = None, None, None
        if accelerator.is_main_process:
            data_transforms = utils.get_data_augmentation(args)
            image_dataset = utils.load_image_dataset(args)
            image_dataset.set_format("torch")

            train_dataset, val_dataset, _ = utils.preprocess_train_eval_data(image_dataset, data_transforms)

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
            "loss": HammingDistance(**metric_params),
            "auc": AUROC(**metric_params_clone),
            "acc": Accuracy(**metric_params),
            "f1": F1Score(**metric_params),
            "recall": Recall(**metric_params),
            "precision": Precision(**metric_params),
            "cm": ConfusionMatrix(**{"task": task, "num_classes": num_classes})
        })

        roc_metric = ROC(**{"task": task, "num_classes": num_classes}).to(device)

        train_metrics = metric_collection.to(device)
        val_metrics = metric_collection.to(device)

        model = utils.get_pretrained_model(args, model_name=args.model_name, num_classes=num_classes)

        model = accelerator.prepare_model(model)

        optimizer = utils.get_optimizer(args, model)
        lr_scheduler = utils.get_lr_scheduler(args, optimizer, len(train_loader))

        train_loader, val_loader = accelerator.prepare_data_loader(
            train_loader), accelerator.prepare_data_loader(val_loader)
        optimizer, lr_scheduler = accelerator.prepare_optimizer(optimizer), accelerator.prepare_scheduler(
            lr_scheduler)

        start_epoch = 0
        best_f1 = 0.0
        best_results = {}

        loaded_checkpoint = train.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                with accelerator.main_process_first():
                    checkpoint = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"), map_location="cpu")
                    model.load_state_dict(checkpoint["model"])
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    for param_group in optimizer.param_groups:
                        if "lr" in config:
                            param_group["lr"] = config["lr"]
                        if "weight_decay" in config:
                            param_group["weight_decay"] = config["weight_decay"]

                    start_epoch = checkpoint["epoch"] + 1
                    best_f1 = checkpoint["best_f1"]
                    best_results = checkpoint["best_results"]

                    if start_epoch == args.epochs:
                        accelerator.print("Training completed")
                    else:
                        accelerator.print(f"Resuming training from epoch {start_epoch}\n")

        for epoch in range(start_epoch, args.epochs):
            train_metrics.reset()

            train_one_epoch(args, epoch, classes, train_loader, model, optimizer, criterion, train_metrics,
                            accelerator)

            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()

            val_metrics.reset()
            roc_metric.reset()

            total_val_metrics, total_roc_metric = evaluate(classes, val_loader, model, val_metrics, roc_metric,
                                                           accelerator)

            val_results = {key: value.detach().tolist() if key == "cm" else round(value.item(), 4) for key, value in
                           total_val_metrics.items()}

            if args.prune:
                parameters_to_prune = utils.prune_model(model, args.pruning_rate)
                utils.remove_pruning_reparam(parameters_to_prune)

            with tempfile.TemporaryDirectory() as tempdir:
                accelerator.wait_for_everyone()
                if val_results["f1"] >= best_f1:
                    best_f1 = val_results["f1"]
                    best_results = val_results

                    accelerator.save(
                        {
                            "model": accelerator.get_state_dict(model)
                        },
                        os.path.join(tempdir, "best_model.pth"))

                metrics = {
                    "auc": best_results["auc"],
                    "f1": best_results["f1"],
                    "precision": best_results["precision"]
                    "recall": best_results["recall"],
                }
                accelerator.save(
                    {
                        "epoch": epoch,
                        "best_f1": best_f1,
                        "model": accelerator.get_state_dict(model),
                        "optimizer": accelerator.get_state_dict(optimizer),
                        "lr_scheduler": accelerator.get_state_dict(lr_scheduler),
                        "best_results": best_results,
                    },
                    os.path.join(tempdir, "checkpoint.pt"),
                )

                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
        accelerator.print("Finished Training")

    inner_main_loop()


def main(args):
    config = {}
    hyperparam_mutations = {}

    if args.tune_opt:
        config["lr"] = tune.qloguniform(5e-3, 1e-1, 5e-4)
        config["weight_decay"] = tune.qloguniform(5e-3, 1e-1, 5e-4)

        hyperparam_mutations["lr"] = [5e-3, 1e-1]
        hyperparam_mutations["weight_decay"] = [5e-3, 1e-1]

    if args.tune_batch_size:
        config["batch_size"] = tune.choice([16, 32, 64, 128])
        hyperparam_mutations["batch_size"] = [16, 128]

    if args.tune_smoothing:
        config["smoothing"] = tune.choice([0.05, 0.1, 0.15])
        hyperparam_mutations["smoothing"] = [0.05, 0.15]

    if args.tune_sched:
        config["warmup_epochs"] = tune.randint(5, 20)
        config["warmup_decay"] = tune.choice([0.1, 0.01, 0.001])

        hyperparam_mutations["warmup_epochs"] = [5, 20]
        hyperparam_mutations["warmup_decay"] = [0.1, 0.001]
        if args.sched_name == "step":
            config["step_size"] = tune.randint(10, 30)
            config["gamma"] = tune.uniform(0.01, 0.1)
            hyperparam_mutations["step_size"] = [10, 30]
            hyperparam_mutations["gamma"] = [0.01, 0.1]
        elif args.sched_name == "cosine":
            config["eta_min"] = tune.qloguniform(5e-3, 1e-1, 5e-4)
            hyperparam_mutations["eta_min"] = [5e-3, 1e-1]
        elif args.sched_name == "cosine_wr":
            config["t0"] = tune.randint(2, 25)
            config["eta_min"] = tune.qloguniform(5e-3, 1e-1, 5e-4)
            hyperparam_mutations["t0"] = [2, 25]
            hyperparam_mutations["eta_min"] = [5e-3, 1e-1, 5e-4]
        elif args.sched_name == "one_cycle":
            config["max_lr"] = tune.qloguniform(5e-3, 1e-1, 5e-4)
            hyperparam_mutations["max_lr"] = [5e-3, 1e-1]

    if args.tune_dropout:
        config["dropout"] = tune.choice([0.0, 0.1, 0.2, 0.3, 0.4])
        hyperparam_mutations["dropout"] = [0.0, 0.4]

    if args.tune_aug_type:
        config["aug_type"] = tune.choice(["augmix", "rand", "trivial"])

    if args.tune_interpolation:
        config["interpolation"] = tune.choice(["nearest", "bilinear", "bicubic"])

    if args.tune_mag_bins:
        config["mag_bins"] = tune.qrandint(4, 32, 4)
        hyperparam_mutations["mag_bins"] = [4, 32]

    if args.tune_mixup:
        config["mixup_alpha"] = tune.quniform(0.1, 0.6, 0.1)
        hyperparam_mutations["mixup_alpha"] = [0.1, 0.6]

    if args.tune_cutmix:
        config["cutmix_alpha"] = tune.quniform(0.1, 0.6, 0.1)
        hyperparam_mutations["cutmix_alpha"] = [0.1, 0.6]

    if args.tune_fgsm:
        config["epsilon"] = tune.quniform(0.01, 0.1, 0.01)
        hyperparam_mutations["epsilon"] = [0.01, 0.1]

    if args.tune_prune:
        config["pruning_rate"] = tune.quniform(0.1, 1.0, 0.1)
        hyperparam_mutations["pruning_rate"] = [0.1, 1.0]

    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    scheduler = None
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=args.epochs,
            grace_period=1,
            reduction_factor=2
        )

    perturbation_interval = 300.0
    if args.scheduler == "pbt":
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=perturbation_interval,
            quantile_fraction=0.5,
            resample_probability=0.5,
            hyperparam_mutations=hyperparam_mutations,
            synch=True,
        )

    if args.scheduler == "pb2":
        scheduler = PB2(
            time_attr="training_iteration",
            perturbation_interval=perturbation_interval,
            quantile_fraction=0.5,
            hyperparam_bounds=hyperparam_mutations,
            synch=True,
        )

    search_algo = TuneBOHB(metric=args.sorting_metric, mode="max")

    storage_path = os.path.expanduser(f"~/{args.output_dir}")
    exp_dir = os.path.join(storage_path, args.experiment_name)

    if tune.Tuner.can_restore(exp_dir):
        tuner = tune.Tuner.restore(
            path=exp_dir,
            trainable=tune.with_resources(
                tune.with_parameters(partial(tune_classifier, args=args)),
                resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
            ),
            resume_unfinished=True,
            resume_errored=True,
            param_space=config,
        )
    else:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(partial(tune_classifier, args=args)),
                resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric=args.sorting_metric,
                mode="max",
                scheduler=scheduler,
                num_samples=args.num_samples,
                search_alg=search_algo if args.scheduler == "asha" else None,
                max_concurrent_trials=10,
            ),
            run_config=train.RunConfig(
                name=args.experiment_name,
                storage_path=storage_path,
                log_to_file=True,
                stop={
                    args.sorting_metric: 0.9998,
                    "training_iteration": args.epochs,
                },
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_score_attribute=args.sorting_metric,
                    checkpoint_score_order="max",
                    num_to_keep=3
                ),
                failure_config=train.FailureConfig(max_failures=3),
                progress_reporter=tune.CLIReporter(metric_columns=["auc", "f1", "precision", "recall"])
            ),
            param_space=config,
        )
    results = tuner.fit()

    return results


def get_args():
    """
    Parse and return the command line arguments.

    Returns:
        argparse.Namespace: A namespace containing parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Classification Hyperparameter Tuning")

    # General Configuration
    parser.add_argument(
        "--experiment_name",
        required=True,
        default="Experiment_1",
        type=str,
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
        default="",
        type=str,
        help="Path to a JSON file containing keyword arguments (kwargs) specific to a HuggingFace dataset."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The directory where output files, such as trained models and evaluation results, will be saved."
    )
    parser.add_argument(
        "--seed",
        default=999333666,
        type=int,
        help="Set the random seed for reproducibility of training results."
    )
    parser.add_argument(
        "--mixed_precision",
        default=None,
        type=str,
        help="Enable to use mixed precision, and the type of precision to use",
        choices=["no", "fp16", "bf16", "fp8"]
    )

    # Model Configuration
    parser.add_argument(
        "--feat_extract",
        action="store_true",
        help="Enable feature extraction during training when using pretrained models."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Specify the name of the model from the TIMM library."
    )

    # Training Configuration
    # FGSM Adversarial Training
    parser.add_argument(
        "--fgsm",
        action="store_true",
        help="Enable FGSM (Fast Gradient Sign Method) adversarial training to enhance model robustness."
    )
    parser.add_argument(
        "--epsilon",
        default=0.03,
        type=float,
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
        default=0.25,
        type=float,
        help="Set the pruning rate to control the extent of pruning applied to the model."
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
        default="auto",
        type=str,
        help="The type of data augmentation to use.",
        choices=["augmix", "auto", "rand", "trivial"]
    )
    parser.add_argument(
        "--interpolation",
        default="bicubic",
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
        default=1.0,
        type=float,
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
        default=1.0,
        type=float,
        help="Set the cutmix hyperparameter alpha to control the interpolation factor."
    )
    parser.add_argument(
        '--rand_erase_prob',
        default=0.0,
        type=float,
        help='Probability of applying Random Erase augmentation.'
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
        default=0.2,
        type=float,
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
        default=0.1,
        type=float,
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
        default=5,
        type=int,
        help="The number of iterations for the first restart in learning rate scheduling strategies."
    )

    # Evaluation Metrics and Testing
    parser.add_argument(
        "--sorting_metric",
        default="f1",
        type=str,
        help="The metric by which the model results will be sorted.",
        choices=["accuracy", "auc", "f1", "precision", "recall"]
    )

    # Hparams tuning arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to search for hyperparameters"
    )
    parser.add_argument(
        "--cpus_per_trial",
        type=int,
        default=8,
        help="Number of CPUs per trial"
    )
    parser.add_argument(
        "--gpus_per_trial",
        type=int,
        default=0,
        help="Number of GPUs per trial"
    )
    parser.add_argument(
        "--search_algo",
        type=str,
        default="bohb",
        help="Hyperparameter search algorithm"
    )

    parser.add_argument(
        "--scheduler",
        default="asha",
        type=str,
        help="Type of optimization algorithm to use. Available are Population Based Training (PBT), Population Based "
             "Bandits (PB2), and Adaptive Successive Halving",
        choices=["pbt", "pb2", "asha"]
    )

    parser.add_argument(
        "--tune_smoothing",
        action="store_true",
        help="whether to tune label smoothing hyperparameter"
    )

    parser.add_argument(
        "--tune_dropout",
        action="store_true",
        help="whether to tune dropout rate hyperparameter"
    )
    parser.add_argument(
        "--tune_mixup",
        action="store_true",
        help="whether to tune mixup"
    )
    parser.add_argument(
        "--tune_cutmix",
        action="store_true",
        help="whether to tune cutmix"
    )
    parser.add_argument(
        "--tune_fgsm",
        action="store_true",
        help="whether to tune fgsm"
    )
    parser.add_argument(
        "--tune_prune",
        action="store_true",
        help="whether to tune pruning"
    )
    parser.add_argument(
        "--tune_aug_type",
        action="store_true",
        help="whether to tune augmentation type hyperparameter"
    )
    parser.add_argument(
        "--tune_mag_bins",
        action="store_true",
        help="whether to tune magnitude bins hyperparameter"
    )
    parser.add_argument(
        "--tune_batch_size",
        action="store_true",
        help="whether to tune batch size hyperparameter"
    )
    parser.add_argument(
        "--tune_opt",
        action="store_true",
        help="whether to tune optimizer hyperparameters"
    )
    parser.add_argument(
        "--tune_sched",
        action="store_true",
        help="whether to tune learning rate schedule hyperparameters"
    )
    parser.add_argument(
        "--tune_interpolation",
        action="store_true",
        help="whether to tune image interpolation method hyperparameter"
    )

    return parser.parse_args()


if __name__ == "__main__":
    torch.jit.enable_onednn_fusion(True)

    warnings.filterwarnings("ignore")
    cfgs = get_args()
    cfgs.module = None
    cfgs.model_size = None

    set_seed(cfgs.seed)

    tune_results = main(cfgs)
    best_result = tune_results.get_best_result(metric=cfgs.sorting_metric, mode="max")

    print(f"\nBest trial config: {best_result.config}")
    print(f"Best trial final validation metrics: {[{metric: best_result.metrics[metric] for metric in ['auc', 'f1', 'precision', 'recall']}][0]}")

