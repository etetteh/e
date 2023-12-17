import argparse
import os
import warnings
import subprocess

from accelerate import (
    Accelerator,
    FullyShardedDataParallelPlugin
)
from accelerate.utils import set_seed
# noinspection PyProtectedMember
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import pandas as pd

import torch
import utils
from explain import explain_model
from train import main
from tune import main as tune_main
from inference import run_inference
import streamlit as st


def get_input_config(key):
    cfgs = argparse.Namespace()

    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        cfgs.experiment_name = st.text_input("Experiment Name",
                                             help="This command allows you to specify the name of the MLflow "
                                                  "experiment. It helps organize and categorize the experimental runs.",
                                             key=f"exp_{key}")

        cfgs.dataset = st.text_input("Dataset Path",
                                     help="Use this command to provide the path to the dataset directory or the name "
                                          "of a HuggingFace dataset. It defines the data source for model training and "
                                          "evaluation.",
                                     key=f"dataset_{key}")

    with ex_col2:
        cfgs.output_dir = st.text_input("Output Directory",
                                        help="Specify the directory where the output files, such as trained models and "
                                             "evaluation results, will be saved.",
                                        key=f"output_{key}")

        cfgs.dataset_kwargs = st.text_input("Dataset Kwargs JSON Path",
                                            help="If needed, you can use this command to point to a JSON file "
                                                 "containing keyword arguments (kwargs) specific to a HuggingFace "
                                                 "dataset.",
                                            key=f"datasets_kwargs_{key}")

    cfgs.aug_type = st.selectbox("Data Augmentation Type", ["rand", "trivial", "augmix"],
                                 help="Choose the type of data augmentation to use",
                                 key=f"aug_type_{key}")
    cfgs.interpolation = st.selectbox("Interpolation", ["bilinear", "nearest", "bicubic"],
                                      help="Choose the type of interpolation method to use",
                                      key=f"interpolation_{key}")

    if key == "tune":
        model_name = st.text_input("Model Name",
                                   help="Use this to specify the name of the model you want to use from the TIMM.",
                                   key=f"model_name_{key}")
        cfgs.model_name = model_name.split()
    else:
        model_selection = st.radio("Select Model Configuration",
                                   ["Model Name", "Model Size", "Module"],
                                   captions=["Specify model name or list of model names",
                                             "Specify model size. e.g. base",
                                             "Specify a model submodule. e.g. deit"],
                                   key=f"model_{key}",
                                   horizontal=True
                                   )
        cfgs.module = None
        cfgs.model_name = None
        cfgs.model_size = None

        if model_selection == "Module":
            modules = ['beit', 'convnext', 'deit', 'resnet', 'vision_transformer', 'efficientnet', 'xcit', 'regnet',
                       'nfnet', 'metaformer', 'fastvit', 'efficientvit_msra', "Other"]
            module = st.selectbox("Select Models' Submodule", modules,
                                  help="If you want to select a specific model submodule, such as 'resnet' or 'deit', "
                                       "you can use this command to make that choice. It's not compatible with the "
                                       "--model_size or --model_name commands")
            if module == "Other":
                cfgs.module = st.text_input("Enter your preferred submodule")
            else:
                cfgs.module = module

            if key == "train":
                filter_module = st.text_input(
                    f"Optionally, Exclude Model Size(s)",
                    help="Use this to specify the model size(s) to exclude. "
                         "Choose from 'enormous', 'huge', 'giant', 'large'",
                    key=f"filter_module_{key}",

                )
                cfgs.filter_module = filter_module.split()

        elif model_selection == "Model Size":
            cfgs.model_size = st.selectbox(
                "Select Model Size", ["nano", "tiny", "small", "base", "large", "giant"],
                help="If you prefer to specify the size of the model, you can use this "
                     "command. It's not used when Model Name or Module are specified specified."
            )
        elif model_selection == "Model Name":
            model_name = st.text_input(
                "Model Name(s)",
                help="Use this to specify the name of the model(s) you want to use from the TIMM"
                     " library. It's not compatible with the Model Size or Module commands.",
                key=f"model_name_{key}",
            )
            cfgs.model_name = model_name.split()

    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        optimizers = ["lion", "madgradw", "adamw", "radabelief", "adafactor", "novograd", "lars", "lamb", "rmsprop",
                      "sgdp", "Other"]
        opt_name = st.selectbox("Optimizer Name", optimizers, help="Choose the optimizer for the training process.",
                                key=f"opt_name_{key}")
        if opt_name == "Other":
            cfgs.opt_name = st.text_input("Enter your preferred optimizer")
        else:
            cfgs.opt_name = opt_name

    with opt_col2:
        cfgs.sched_name = st.selectbox("Learning Rate Scheduler", ["step", "cosine", "cosine_wr", "one_cycle"],
                                       help="Choose the learning rate scheduler strategy",
                                       key=f"sched_name_{key}")

    st.subheader("Advanced Configuration")
    column1, column2 = st.columns(2)

    with column1:
        cfgs.fgsm = st.toggle("Enable FGSM Adversarial Training",
                              help="This flag allows you to enable FGSM (Fast Gradient Sign Method) adversarial "
                                   "training to enhance model robustness.",
                              key=f"fgsm_{key}")
        if cfgs.fgsm:
            cfgs.epsilon = st.number_input("FGSM epsilon Value", value=0.03,
                                           help="If FGSM adversarial training is enabled, you can set the epsilon value"
                                                "for the FGSM attack using this command.")

        cfgs.mixup = st.toggle("Enable Mixup",
                               help="Include this flag to enable mixup augmentation, which enhances training by mixing "
                                    "pairs of samples.",
                               key=f"mixup_{key}")
        if cfgs.mixup:
            cfgs.mixup_alpha = st.number_input("Mixup Alpha", value=1.0,
                                               help="Set the mixup hyperparameter alpha to control the interpolation "
                                                    "factor.")

        cfgs.cutmix = st.toggle("Enable Cutmix",
                                help="Enable cutmix augmentation, which combines patches from different images to "
                                     "create"
                                     "new training samples.",
                                key=f"cutmix_{key}")
        if cfgs.cutmix:
            cfgs.cutmix_alpha = st.number_input("Cutmix Alpha", value=1.0,
                                                help="Set the cutmix hyperparameter alpha to control the interpolation "
                                                     "factor.")

        cfgs.avg_ckpts = st.toggle("Enable Checkpoint Averaging",
                                   help="When enabled, this flag triggers checkpoint averaging during training. It "
                                        "helps"
                                        "stabilize the training process.",
                                   key=f"avg_ckpts_{key}")

    with column2:
        cfgs.feat_extract = st.toggle("Enable Feature Extraction",
                                      help="By including this flag, you can enable feature extraction during training, "
                                           "which is useful when using pretrained models.",
                                      key=f"feat_extract_{key}")
        cfgs.grayscale = st.toggle("Use Grayscale Images",
                                   help="If needed, use this flag to indicate that grayscale images should be used "
                                        "during"
                                        "training.",
                                   key=f"grayscale_{key}")

        cfgs.prune = st.toggle("Enable Pruning",
                               help="Include this flag to enable pruning during training, which helps reduce model "
                                    "complexity and size.",
                               key=f"prune_{key}")
        if cfgs.prune:
            cfgs.pruning_rate = st.number_input("Pruning Rate", value=0.25,
                                                help="Set the pruning rate to control the extent of pruning applied to "
                                                     "the model.")

        cfgs.to_onnx = st.toggle("Convert All Models to ONNX Format",
                                 help="Include this flag if you want to convert the trained model(s) to ONNX format. If"
                                      "not used, only the best model will be converted.",
                                 key=f"to_onnx_{key}")

    with st.expander("More configurations"):
        misc_col1, misc_col2, misc_col3, misc_col4 = st.columns(4)

        with misc_col1:
            cfgs.epochs = st.number_input("Number of Epochs", value=100,
                                          help="Set the number of training epochs, determining how many times the "
                                               "entire dataset will be iterated.",
                                          key=f"epochs_{key}")

            cfgs.batch_size = st.number_input("Batch Size", value=16,
                                              help=" Define the batch size for both training and evaluation stages",
                                              key=f"batch_size_{key}")

            cfgs.num_workers = st.number_input("Number of Workers", value=4,
                                               help="Specify the number of workers for training and evaluation.",
                                               key=f"num_workers_{key}")

            cfgs.num_ckpts = st.number_input("Number of Checkpoints", min_value=1, value=1,
                                             help="Set the number of best checkpoints to save when checkpoint "
                                                  "averaging is active. It determines how many checkpoints are "
                                                  "averaged.",
                                             key=f"num_ckpts_{key}")

        with misc_col2:
            cfgs.lr = st.number_input("Initial Learning Rate (lr)", value=0.001,
                                      help=" Specify the initial learning rate for the optimizer.",
                                      key=f"lr_{key}")

            cfgs.wd = st.number_input("Weight Decay", value=1e-4,
                                      help="Set the weight decay (L2 regularization) for the optimizer.",
                                      key=f"wd_{key}")

            cfgs.dropout = st.number_input("Dropout Rate", value=0.2,
                                           help="Define the dropout rate for the classifier head of the model.",
                                           key=f"dropout_{key}")

            cfgs.label_smoothing = st.number_input("Label Smoothing", value=0.1,
                                                   help="Set the amount of label smoothing to use during training.",
                                                   key=f"label_smoothing_{key}")

        with misc_col3:
            cfgs.crop_size = st.number_input("Crop Size", value=224,
                                             help="Define the size to which input images will be cropped.",
                                             key=f"crop_size_{key}")

            cfgs.val_resize = st.number_input("Validation Resize", value=256,
                                              help="Specify the size to which validation images will be resized.",
                                              key=f"val_resize_{key}")

            cfgs.hflip = st.number_input("Horizontal Flip Probability", value=0.5,
                                         help="Define the probability of randomly horizontally flipping the input "
                                              "data.",
                                         key=f"hflip_{key}")

            cfgs.mag_bins = st.number_input("Magnitude bins", value=31,
                                            help="Set the number of magnitude bins for augmentation-related "
                                                 "operations.",
                                            key=f"mag_bins_{key}")

        with misc_col4:
            cfgs.eta_min = st.number_input("Minimum Learning Rate (eta_min)", value=1e-4,
                                           help="Define the minimum learning rate that the scheduler can reach.",
                                           key=f"eta_min_{key}")

            cfgs.max_lr = st.number_input("Maximum Learning Rate (max_lr)", value=0.1,
                                          help="Set the maximum learning rate when using cyclic learning rate "
                                               "scheduling.",
                                          key=f"max_lr_{key}")

            cfgs.step_size = st.number_input("Step Size", value=30,
                                             help="Set the step size for learning rate adjustments in certain "
                                                  "scheduler strategies.",
                                             key=f"step_size_{key}")

            cfgs.warmup_epochs = st.number_input("Warmup Epochs", value=5,
                                                 help="Specify the number of epochs for the warmup phase of learning "
                                                      "rate scheduling.",
                                                 key=f"warmup_epochs_{key}")

            cfgs.warmup_decay = st.number_input("Warmup Decay", value=0.1,
                                                help="Set the decay rate for the learning rate during the warmup "
                                                     "phase.",
                                                key=f"warmup_decay_{key}")

            cfgs.gamma = st.number_input("Gamma", value=0.1,
                                         help="Set the gamma parameter used in certain learning rate "
                                              "scheduling strategies.",
                                         key=f"gamma_{key}")

            cfgs.t0 = st.number_input("First Restart Iterations (t0)", value=5,
                                      help="Specify the number of iterations for the first restart in learning rate "
                                           "scheduling strategies.",
                                      key=f"t0_{key}")

        cfgs.seed = st.number_input("Random Seed", value=999333666,
                                    help="Set the random seed for reproducibility of training results.",
                                    key=f"seed_{key}")

        cfgs.sorting_metric = st.selectbox("Sorting Metric", ["f1", "auc", "accuracy", "precision", "recall"],
                                           help="Choose the metric by which the model results will be sorted.",
                                           key=f"sorting_metric_{key}")

    return cfgs


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    bash_command = "mlflow ui"
    sub_process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    with st.sidebar:
        st.title("Low Code Image Classification")
        st.write("The best image classification project yet")

        st.info("Author: Enoch Tetteh")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Training", "Model Evaluation Results", "Model Hyperparameter Tuning",
                                            "Inference", "Model Explanation"])

    with tab1:
        st.header("Image Classification Training")
        warnings.filterwarnings("ignore")

        if torch.cuda.is_available():
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
            )
        else:
            fsdp_plugin = None

        accelerator_var = Accelerator(
            even_batches=True,
            gradient_accumulation_steps=2,
            mixed_precision="fp16",
            fsdp_plugin=fsdp_plugin
        )

        train_cfgs = get_input_config("train")

        train_cfgs.test_only = st.toggle("Enable Test Only",
                                         help="When enabled, this flag indicates that you want to perform"
                                              " testing on the test split only, skipping training.")

        method = "**Run Evaluation**" if train_cfgs.test_only else "**Start Training**"
        status = "Evaluation in progress" if train_cfgs.test_only else "Training in progress"
        update_label = "Model evaluation complete!" if train_cfgs.test_only else "Model training complete!"

        if st.button(method):
            with st.status(status, expanded=True) as status:

                set_seed(train_cfgs.seed)

                if not os.path.isdir(train_cfgs.output_dir):
                    os.makedirs(train_cfgs.output_dir, exist_ok=True)
                    accelerator_var.print(f"Output directory created: {os.path.abspath(train_cfgs.output_dir)}")
                else:
                    accelerator_var.print(
                        f"Output directory already exists at: {os.path.abspath(train_cfgs.output_dir)}")

                if train_cfgs.model_name is not None:
                    train_cfgs.models = sorted(train_cfgs.model_name)
                else:
                    train_cfgs.models = sorted(utils.get_matching_model_names(train_cfgs))

                train_cfgs.lr *= accelerator_var.num_processes

                main(train_cfgs, accelerator_var)

                status.update(label=update_label, state="complete", expanded=False)

        st.divider()

        jsonl_file = None
        if train_cfgs.test_only:
            st.subheader("Test Results")
            if os.path.isfile(os.path.join(train_cfgs.output_dir, "test_performance_metrics.jsonl")):
                jsonl_file = os.path.join(train_cfgs.output_dir, "test_performance_metrics.jsonl")
            else:
                st.write("Click on **Start Training** button with **Enable Test Only** active, to obtain performance on"
                         " test data")
        else:
            st.subheader("Training Results")
            if os.path.isfile(os.path.join(train_cfgs.output_dir, "performance_metrics.jsonl")):
                jsonl_file = os.path.join(train_cfgs.output_dir, "performance_metrics.jsonl")

        if jsonl_file is not None:
            st.button("Clear results", type="primary")
            if st.button("Show results", type="primary"):
                df = pd.read_json(jsonl_file, lines=True)
                st.dataframe(df)

    with tab2:
        st.header("Evaluation Results")

        args = argparse.Namespace()
        args.output_dir = st.text_input("Enter Directory of Trained Model")

        if os.path.isdir(args.output_dir):
            st.divider()
            results_list = utils.read_json_lines_file(os.path.join(args.output_dir, "performance_metrics.jsonl"))
            best_compare_model_name = os.path.join(args.output_dir, results_list[0]["model"])
            st.subheader("Confusion Matrix")
            if os.path.isfile(os.path.join(best_compare_model_name, "confusion_matrix.png")):
                st.image(os.path.join(best_compare_model_name, "confusion_matrix.png"))

            st.subheader("ROC AUC Curve")
            if os.path.isfile(os.path.join(best_compare_model_name, "roc_curve.png")):
                st.image(os.path.join(best_compare_model_name, "roc_curve.png"))

    with tab3:
        st.header("Image Classification Hyperparameter Tuning")
        torch.jit.enable_onednn_fusion(True)

        warnings.filterwarnings("ignore")
        tune_cfgs = get_input_config("tune")
        tune_cfgs.module = None
        tune_cfgs.model_size = None

        set_seed(tune_cfgs.seed)

        st.subheader("Required Settings")
        set_col1, set_col2 = st.columns(2)

        with set_col1:
            tune_cfgs.search_algo = st.text_input("Hyperparameter Search Algorithm", value="bohb",
                                                  help="Hyperparameter search algorithm")
            tune_cfgs.gpus_per_trial = st.number_input("Number of GPUs per Trial", value=0,
                                                       help="Number of GPUs per trial")

        with set_col2:
            tune_cfgs.num_samples = st.number_input("Number of Samples to Search for Hyperparameters", value=16,
                                                    help="Number of samples to search for hyperparameters")
            tune_cfgs.cpus_per_trial = st.number_input("Number of CPUs per Trial", value=8,
                                                       help="Number of CPUs per trial")

        st.subheader("Optimization Algorithms (Schedulers)")
        scheduler = st.radio(
            "Select your optimization algorithm (A.K.A scheduler)",
            ["Use ASHA", "Use PBT", "Use PB2"],
            captions=[
                "Use Async Successive Halving (ASHA) algorithm",
                "Use Population Based Training (PBT) algorithm",
                "Use Population Based Bandit (PB2) algorithm"
            ],
            horizontal=True
        )

        tune_cfgs.asha = tune_cfgs.pbt = tune_cfgs.pb2 = False

        if scheduler == "Use ASHA":
            tune_cfgs.asha = True
        elif scheduler == "Use PBT":
            tune_cfgs.pbt = True
        elif scheduler == "Use PB2":
            tune_cfgs.pb2 = True

        st.subheader("Tuning Options")
        col1, col2 = st.columns(2)

        with col1:
            tune_cfgs.tune_mixup = st.checkbox("Tune Mixup probability Hyperparameter", help="Whether to tune mixup")
            tune_cfgs.tune_cutmix = st.checkbox("Tune CutMix probability Hyperparameter", help="Whether to tune cutmix")
            tune_cfgs.tune_fgsm = st.checkbox("Tune FGSM epsilon Hyperparameter", help="Whether to tune fgsm")
            tune_cfgs.tune_prune = st.checkbox("Tune Pruning rate Hyperparameter", help="Whether to tune pruning")
            tune_cfgs.tune_aug_type = st.checkbox("Tune Augmentation Type Hyperparameter",
                                                  help="Whether to tune augmentation type hyperparameter")
            tune_cfgs.tune_dropout = st.checkbox("Tune Dropout rate Hyperparameter",
                                                 help="Whether to tune dropout rate hyperparameter")

        with col2:
            tune_cfgs.tune_smoothing = st.checkbox("Tune Label Smoothing rate Hyperparameter",
                                                   help="Whether to tune label smoothing hyperparameter")
            tune_cfgs.tune_mag_bins = st.checkbox("Tune Magnitude Bins Hyperparameter",
                                                  help="Whether to tune magnitude bins hyperparameter")
            tune_cfgs.tune_batch_size = st.checkbox("Tune Batch Size Hyperparameter",
                                                    help="Whether to tune batch size hyperparameter")
            tune_cfgs.tune_opt = st.checkbox("Tune Optimizer Hyperparameters",
                                             help="Whether to tune the Learning rate and Weight decay")
            tune_cfgs.tune_sched = st.checkbox("Tune Learning Rate Schedule Hyperparameters",
                                               help="Whether to tune learning rate schedule hyperparameters")
            tune_cfgs.tune_interpolation = st.checkbox("Tune Image Interpolation Method Hyperparameter",
                                                       help="Whether to tune image interpolation method hyperparameter")

        if st.button("**Tune Hyperparameters**"):
            with st.status("Hyperparameter tuning in progress", expanded=True) as status:
                results = tune_main(tune_cfgs)
                status.update(label="Hyperparameter tuning complete!", state="complete", expanded=False)

        st.divider()

        st.subheader("Hyperparameter Tuning Results")

        st.button("Clear inference results", type="primary")
        if st.button("Show hyperparameters tuning results", type="primary"):
            best_result = results.get_best_result(metric=tune_cfgs.sorting_metric, mode="max")
            st.write(f"\nBest trial config: {best_result.config}")
            st.write(f"Best trial final validation loss: {best_result.metrics['loss']}")
            st.write(f"Best trial final validation {tune_cfgs.sorting_metric}: "
                     f"{best_result.metrics[tune_cfgs.sorting_metric]}")

    with tab4:
        st.header("Model Inference")

        st.write("Enter the values for the arguments:")

        infer_args = argparse.Namespace()

        infer_args.output_dir = st.text_input("Inference Output Directory", help="Specify the directory where the "
                                                                                 "output files will be saved, "
                                                                                 "such as classification results")
        infer_args.onnx_model_path = st.text_input("ONNX Model Path",
                                                   help="Provide the path to the ONNX model file that you want to use "
                                                        "for inference.")
        infer_args.img_path = st.text_input("Image Path",
                                            help="Specify the path to a single image or a directory containing images "
                                                 "that you want to classify.")

        da_col1, da_col2 = st.columns(2)

        with da_col1:
            infer_args.dataset_dir_or_classes_file = st.text_input("Dataset Directory or Classes File",
                                                                   help="Provide the path to either the directory "
                                                                        "containing the dataset classes"
                                                                        "or the path to a text file containing class "
                                                                        "names. This helps map the model's output to"
                                                                        "human-readable class names.")

        with da_col2:
            infer_args.dataset_kwargs = st.text_input("Dataset Kwargs (Path to JSON file)",
                                                      help="If necessary, you can provide the path to a JSON file "
                                                           "containing keyword arguments (kwargs) specific to a "
                                                           "HuggingFace dataset.")

        aug_col1, aug_col2 = st.columns(2)

        with aug_col1:
            infer_args.crop_size = st.number_input("Inference Image Crop Size", value=224,
                                                   help="Define the size to which input images will be cropped during "
                                                        "inference.")

        with aug_col2:
            infer_args.val_resize = st.number_input("Inference Image Resize", value=256,
                                                    help="Specify the size to which validation images will be resized "
                                                         "during inference.")

        infer_args.grayscale = st.toggle("Use Grayscale Images", help="Use this flag if you want to use grayscale "
                                                                      "images during inference.")

        if st.button("**Run Inference**"):
            with st.status("Model inference in progress", expanded=True) as status:
                warnings.filterwarnings("ignore", category=UserWarning)

                if not os.path.isdir(infer_args.output_dir):
                    os.makedirs(infer_args.output_dir, exist_ok=True)

                result = run_inference(infer_args)
                utils.write_dictionary_to_json(dictionary=result,
                                               file_path=f"{infer_args.output_dir}/inference_results.json")

                status.update(label="Model inference complete!", state="complete", expanded=False)

        st.divider()

        st.subheader("Inference Results")
        st.button("Clear inference results", type="primary", key="results_tune")
        if st.button("Show inference results", type="primary"):
            if os.path.isfile(os.path.join(infer_args.output_dir, "inference_results.json")):
                st.json(utils.read_json_file(os.path.join(infer_args.output_dir, "inference_results.json")))
        else:
            pass

    with tab5:
        st.header("Model Explanation")
        args = argparse.Namespace()

        args.model_output_dir = st.text_input("Trained Model Output Directory",
                                              help="Specifies the output directory where the model is contained")

        data_col1, data_col2 = st.columns(2)
        with data_col1:
            args.dataset = st.text_input("Dataset",
                                         help="Use this command to provide the path to the dataset directory or the "
                                              "name of a HuggingFace dataset. It defines the data source for the "
                                              "explanation")
        with data_col2:
            args.dataset_kwargs = st.text_input("Dataset kwargs",
                                                help="If necessary, you can provide the path to a JSON file containing"
                                                     " keyword arguments (kwargs) specific to a HuggingFace dataset.")

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        with d_col1:
            args.crop_size = st.number_input("Image Crop Size", value=224,
                                             help="Define the size to which input images will be cropped.")

        with d_col2:
            args.batch_size = st.number_input("Batch Size", value=128,
                                              help="Define the batch size for both training and evaluation stages.")

        with d_col3:
            args.num_workers = st.number_input("Number of Workers", value=8,
                                               help="Specify the number of workers for training and evaluation.")

        with d_col4:
            args.dropout = st.number_input("Dropout", value=0.2,
                                           help="Define the dropout rate for the classifier head of the model.")

        args.grayscale = st.toggle("Use Grayscale Images", value=False, key="inference",
                                   help="Use this flag if you want to use grayscale images.")
        args.feat_extract = st.toggle("Enable Feature Extraction", value=True,
                                      help="By including this flag, you can enable feature extraction during"
                                           " training, which is useful when using pretrained models.")

        st.subheader("Explainability Settings")
        args.n_samples = st.number_input("Number of Samples", value=4, help="Specifies the number of samples used for "
                                                                            "model explanation or evaluation.")
        args.max_evals = st.number_input("Max Evaluations", value=777, help="Sets the maximum number of evaluations, "
                                                                            "commonly used for model explanation")
        args.topk = st.number_input("Top K", value=2, help="Indicates the number of top predictions to consider "
                                                           "during model explanation")

        if st.button("**Explain Model**"):
            with st.status("Model explanation in progress", expanded=True) as status:
                explain_model(args)
                status.update(label="Model explanation complete!", state="complete", expanded=False)

        st.divider()

        st.subheader("Explanation Results")
        st.button("Clear explanation results", type="primary")
        if st.button("Show explanation results", type="primary"):
            if os.path.isfile(f"{args.model_output_dir}/explanation.png"):
                st.image(f"{args.model_output_dir}/explanation.png")
        else:
            pass
