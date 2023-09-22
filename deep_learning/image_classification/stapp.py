
import argparse
import os
import warnings
import subprocess

import accelerate
from accelerate import (
    Accelerator,
    DeepSpeedPlugin,
    FullyShardedDataParallelPlugin,
    find_executable_batch_size
)
from accelerate.utils import set_seed

import mlflow
import pandas as pd

import process
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

import utils
from explain import explain_model
from train import main
from inference import run_inference
import streamlit as st


if __name__ == "__main__":

    bash_command = "mlflow ui"
    sub_process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    with st.sidebar:
        st.title("Low Code Image Classification")
        st.write("The best image classification project yet")

        st.info("Author: Enoch Tetteh")

    tab1, tab2, tab3, tab4 = st.tabs(["Model Training", "Evaluation Results", "Inference", "Model Explanation"])
    
    with tab1:
        warnings.filterwarnings("ignore")
    
        deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=2, gradient_clipping=1.0)
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
        accelerator_var = Accelerator(even_batches=True,
                                      gradient_accumulation_steps=2,
                                      mixed_precision="fp16",
                                      deepspeed_plugin=deepspeed_plugin,
                                      fsdp_plugin=fsdp_plugin
                                      )
    
        cfgs = argparse.Namespace()
    
        st.header("Image Classification Training")
        st.write("Enter the values for the arguments:")

        cfgs.experiment_name = st.text_input("Experiment Name", help="This command allows you to specify the name of the MLflow experiment. It helps organize and categorize the experimental runs.")
        cfgs.dataset = st.text_input("Dataset Path", help="Use this command to provide the path to the dataset directory or the name of a HuggingFace dataset. It defines the data source for model training and evaluation.")
        cfgs.dataset_kwargs = st.text_input("Dataset Kwargs JSON Path", help="If needed, you can use this command to point to a JSON file containing keyword arguments (kwargs) specific to a HuggingFace dataset.")
        cfgs.output_dir = st.text_input("Output Directory", key="train", help="Specify the directory where the output files, such as trained models and evaluation results, will be saved.")
        cfgs.seed = st.number_input("Random Seed", value=999333666, help="Set the random seed for reproducibility of training results.")
        cfgs.aug_type = st.selectbox("Data Augmentation Type", ["rand", "trivial", "augmix"], help="Choose the type of data augmentation to use")
        cfgs.interpolation = st.selectbox("Interpolation", ["bilinear", "nearest", "bicubic"], help="Choose the type of interpolation method to use")
        cfgs.hflip = st.number_input("Horizontal Flip Probability", value=0.5, help="Define the probability of randomly horizontally flipping the input data.")
        cfgs.crop_size = st.number_input("Crop Size", value=224, help="Define the size to which input images will be cropped.")
        cfgs.val_resize = st.number_input("Validation Resize", value=256, help="Specify the size to which validation images will be resized.")
                
        model_selection = st.radio("Select Model Configuration", ["Model Name", "Model Size", "Module"])
        cfgs.module = None
        cfgs.model_name = None
        cfgs.model_size = None
    
        if model_selection == "Module":
            modules = ['beit', 'convnext', 'deit', 'resnet', 'vision_transformer', 'efficientnet', 'xcit', 'regnet', 'nfnet', 'metaformer', 'fastvit', 'efficientvit_msra', "Other"]
            module = st.selectbox("Select Model Submodule", modules, help="If you want to select a specific model submodule, such as 'resnet' or 'deit', you can use this command to make that choice. It's not compatible with the --model_size or --model_name commands")
            if module == "Other":
                cfgs.module = st.text_input("Enter your preferred submodule")
            else:
                cfgs.module = module
        elif model_selection == "Model Size":
            cfgs.model_size = st.selectbox("Select Model Size", ["nano", "tiny", "small", "base", "large", "giant"], help="If you prefer to specify the size of the model, you can use this command. It's not used when Model Name or Module are specified specified.")
        elif model_selection == "Model Name":
            model_name = st.text_input("Model Name(s)", help="Use this to specify the name of the model(s) you want to use from the TIMM library. It's not compatible with the Model Size or Module commands.")
            # cfgs.model_name = model_name.split(",") if isinstance(model_name, tuple) else model_name
            cfgs.model_name = model_name.split()
                        
        optimizers = ["lion", "madgradw", "adamw", "radabelief", "adafactor", "novograd", "lars", "lamb", "rmsprop", "sgdp", "Other"]
        opt_name = st.selectbox("Optimizer Name", optimizers, help="Choose the optimizer for the training process.")
        if opt_name == "Other":
            cfgs.opt_name = st.text_input("Enter your preferred optimizer")
        else:
            cfgs.opt_name = opt_name
            
        cfgs.sched_name = st.selectbox("Learning Rate Scheduler", ["step", "cosine", "cosine_wr", "one_cycle"], help="Choose the learning rate scheduler strategy")
        
        cfgs.feat_extract = st.toggle("Enable Feature Extraction", help="By including this flag, you can enable feature extraction during training, which is useful when using pretrained models.")
        cfgs.grayscale = st.toggle("Use Grayscale Images", help="If needed, use this flag to indicate that grayscale images should be used during training.")

        cfgs.prune = st.toggle("Enable Pruning", help=" Include this flag to enable pruning during training, which helps reduce model complexity and size.")
        if cfgs.prune:
            cfgs.pruning_rate = st.number_input("Pruning Rate", value=0.25, help="Set the pruning rate to control the extent of pruning applied to the model.")

        cfgs.avg_ckpts = st.toggle("Enable Checkpoint Averaging",  help="When enabled, this flag triggers checkpoint averaging during training. It helps stabilize the training process.")
                    
        cfgs.fgsm = st.toggle("Enable FGSM Adversarial Training", help="This flag allows you to enable FGSM (Fast Gradient Sign Method) adversarial training to enhance model robustness.")
        if cfgs.fgsm:
            cfgs.epsilon = st.number_input("FGSM epsilon Value", value=0.03, help="If FGSM adversarial training is enabled, you can set the epsilon value for the FGSM attack using this command.")
        
        cfgs.ema = st.toggle("Enable Exponential Moving Average", help="Enabling this flag performs Exponential Moving Average (EMA) during training, which can improve model performance and stability.")
        if cfgs.ema:
            cfgs.ema_steps = st.number_input("EMA Steps", min_value=1, value=32, help="Specify the number of iterations for updating the EMA model.")
            cfgs.ema_decay = st.number_input("EMA Decay", value=0.99998, help="Set the EMA decay factor, which influences the contribution of past model weights.")

        cfgs.mixup = st.toggle("Enable Mixup", help="Include this flag to enable mixup augmentation, which enhances training by mixing pairs of samples.")
        if cfgs.mixup:
            cfgs.mixup_alpha = st.number_input("Mixup Alpha", value=1.0, help="Set the mixup hyperparameter alpha to control the interpolation factor.")
        
        cfgs.cutmix = st.toggle("Enable Cutmix", help="Enable cutmix augmentation, which combines patches from different images to create new training samples.")
        if cfgs.cutmix:
            cfgs.cutmix_alpha = st.number_input("Cutmix Alpha", value=1.0, help="Set the cutmix hyperparameter alpha to control the interpolation factor.")

        cfgs.to_onnx = st.toggle("Convert All Models to ONNX Format", help="Include this flag if you want to convert the trained model(s) to ONNX format. If not used, only the best model will be converted.")

        with st.expander("More configurations"):
            cfgs.num_ckpts = st.number_input("Number of Checkpoints", min_value=1, value=1, help="Set the number of best checkpoints to save when checkpoint averaging is active. It determines how many checkpoints are averaged.")
            cfgs.mag_bins = st.number_input("Magnitude bins", value=31, help="Set the number of magnitude bins for augmentation-related operations.")
            cfgs.batch_size = st.number_input("Batch Size", value=16, help=" Define the batch size for both training and evaluation stages")
            cfgs.num_workers = st.number_input("Number of Workers", value=4, help="Specify the number of workers for training and evaluation.")
            cfgs.epochs = st.number_input("Number of Epochs", value=100, help="Set the number of training epochs, determining how many times the entire dataset will be iterated.")
            cfgs.dropout = st.number_input("Dropout Rate", value=0.2, help="Define the dropout rate for the classifier head of the model.")
            cfgs.label_smoothing = st.number_input("Label Smoothing", value=0.1, help="Set the amount of label smoothing to use during training.")
        
            cfgs.wd = st.number_input("Weight Decay", value=1e-4, help="Set the weight decay (L2 regularization) for the optimizer.")
            
            cfgs.max_lr = st.number_input("Maximum Learning Rate", value=0.1, help="Set the maximum learning rate when using cyclic learning rate scheduling.")
            cfgs.eta_min = st.number_input("Minimum Learning Rate", value=1e-4, help="Define the minimum learning rate that the scheduler can reach.")
            cfgs.lr = st.number_input("Initial Learning Rate", value=0.001, help=" Specify the initial learning rate for the optimizer.")
            cfgs.step_size = st.number_input("Step Size", value=30, help="Set the step size for learning rate adjustments in certain scheduler strategies.")
            cfgs.warmup_epochs = st.number_input("Warmup Epochs", value=5, help="Specify the number of epochs for the warmup phase of learning rate scheduling.")
            cfgs.warmup_decay = st.number_input("Warmup Decay", value=0.1, help="Set the decay rate for the learning rate during the warmup phase.")
            cfgs.gamma = st.number_input("Gamma", value=0.1, help="Set the gamma parameter used in certain learning rate scheduling strategies.")
            
            cfgs.t0 = st.number_input("First Restart Iterations", value=5, help="Specify the number of iterations for the first restart in learning rate scheduling strategies.")
        
            cfgs.sorting_metric = st.selectbox("Sorting Metric", ["f1", "auc", "accuracy", "precision", "recall"], help="Choose the metric by which the model results will be sorted.")
        
        cfgs.test_only = st.toggle("Enable Test Only", help="When enabled, this flag indicates that you want to perform testing on the test split only, skipping training.")

        method = "Run Evaluation" if cfgs.test_only else "Start Training"
        status = "Evaluation in progress" if cfgs.test_only else "Training in progress"
        update_label ="Model evaluation complete!" if cfgs.test_only else "Model training complete!"
        
        if st.button(method):
            with st.status(status, expanded=True) as status:
                
                set_seed(cfgs.seed)
                
                if not os.path.isdir(cfgs.output_dir):
                    os.makedirs(cfgs.output_dir, exist_ok=True)
                    accelerator_var.print(f"Output directory created: {os.path.abspath(cfgs.output_dir)}")
                else:
                    accelerator_var.print(f"Output directory already exists at: {os.path.abspath(cfgs.output_dir)}")
                           
                if cfgs.model_name is not None:
                    cfgs.models = sorted(cfgs.model_name)
                else:
                    cfgs.models = sorted(utils.get_matching_model_names(cfgs))
            
                cfgs.lr *= accelerator_var.num_processes
                
                main(cfgs, accelerator_var)

                status.update(label=update_label, state="complete", expanded=False)
                
        st.divider()

        jsonl_file = None
        if cfgs.test_only:
            st.subheader("Test Results")
            if os.path.isfile(os.path.join(cfgs.output_dir, "test_performance_metrics.jsonl")):
                jsonl_file = os.path.join(cfgs.output_dir, "test_performance_metrics.jsonl")
            else:
                st.write("Click on **Start Training** button with **Enable Test Only** active, to obtain performance on test data")
        else:
            st.subheader("Training Results")
            if os.path.isfile(os.path.join(cfgs.output_dir, "performance_metrics.jsonl")):
                jsonl_file = os.path.join(cfgs.output_dir, "performance_metrics.jsonl")

        if not jsonl_file is None:
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
        st.header("Model Inference")

        st.write("Enter the values for the arguments:")
    
        infer_args = argparse.Namespace()
        
        infer_args.output_dir = st.text_input("Inference Output Directory", help="Specify the directory where the output files will be saved, such as classification results")
        infer_args.onnx_model_path = st.text_input("ONNX Model Path", help="Provide the path to the ONNX model file that you want to use for inference.")
        infer_args.img_path = st.text_input("Image Path", help="Specify the path to a single image or a directory containing images that you want to classify.")
        infer_args.dataset_dir_or_classes_file = st.text_input("Dataset Directory or Classes File", help="Provide the path to either the directory containing the dataset classes "
                            "or the path to a text file containing class names. This helps map the model's output to human-readable class names.")
        infer_args.dataset_kwargs = st.text_input("Dataset Kwargs (Path to JSON file)", help=" If necessary, you can provide the path to a JSON file containing keyword arguments (kwargs) specific to a HuggingFace dataset.")
        infer_args.crop_size = st.number_input("Inference Image Crop Size", value=224, help="Define the size to which input images will be cropped during inference.")
        infer_args.val_resize = st.number_input("Inference Image Resize", value=256, help="Specify the size to which validation images will be resized during inference.")
        infer_args.grayscale = st.toggle("Use Grayscale Images", help="Use this flag if you want to use grayscale images during inference.")
        
        if st.button("Run Inference"):
            with st.status("Model inference in progress", expanded=True) as status:
                warnings.filterwarnings("ignore", category=UserWarning)
        
                if not os.path.isdir(infer_args.output_dir):
                    os.makedirs(infer_args.output_dir, exist_ok=True)
            
                result = run_inference(infer_args)
                utils.write_dictionary_to_json(dictionary=result, file_path=f"{infer_args.output_dir}/inference_results.json")
                
                status.update(label="Model inference complete!", state="complete", expanded=False)
            
        st.divider()

        st.subheader("Inference Results")
        st.button("Clear inference results", type="primary")
        if st.button("Show inference results", type="primary"):
            if os.path.isfile(os.path.join(infer_args.output_dir, "inference_results.json")):
                st.json(utils.read_json_file(os.path.join(infer_args.output_dir, "inference_results.json")))
        else:
            pass

    with tab4:
        st.header("Model Explanation")
        args = argparse.Namespace()
    
        args.model_output_dir = st.text_input("Model Output Directory", help="Specifies the output directory where the model is contained")

        args.dataset = st.text_input("Dataset", help="Use this command to provide the path to the dataset directory or the name of a HuggingFace dataset. It defines the data source for the explanation")
        args.dataset_kwargs = st.text_input("Dataset kwargs", help="If necessary, you can provide the path to a JSON file containing keyword arguments (kwargs) specific to a HuggingFace dataset.")
        
        args.crop_size = st.number_input("Image Crop Size", value=224, help="Define the size to which input images will be cropped.")
        args.batch_size = st.number_input("Batch Size", value=128, help="Define the batch size for both training and evaluation stages.")
        args.num_workers = st.number_input("Number of Workers", value=8, help="Specify the number of workers for training and evaluation.")
        args.n_samples = st.number_input("Number of Samples", value=4, help="Specifies the number of samples used for model explanation or evaluation.")
        args.max_evals = st.number_input("Max Evaluations", value=777, help="Sets the maximum number of evaluations, commonly used for model explanation")
        args.topk = st.number_input("Top K", value=2, help=" Indicates the number of top predictions to consider during model explanation")
        args.dropout = st.number_input("Dropout", value=0.2, help="Define the dropout rate for the classifier head of the model.")
        args.grayscale = st.toggle("Use Grayscale Images", value=False, key="inference", help="Use this flag if you want to use grayscale images ")
        args.feat_extract = st.toggle("Enable Feature Extraction", value=True, help="By including this flag, you can enable feature extraction during training, which is useful when using pretrained models.")

        if st.button("Explain Model"):
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
            
