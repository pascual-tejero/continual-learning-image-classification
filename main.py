import torch
import os
import platform
import shutil
import argparse


from utils.get_dataset_mnist import get_dataset_mnist
from utils.get_dataset_cifar10 import get_dataset_cifar10
from utils.get_dataset_cifar100 import get_dataset_cifar100
from utils.get_dataset_cifar100_alternative_dist import get_dataset_cifar100_alternative_dist
from utils.save_global_results import save_global_results

from methods.naive_training import naive_training
from methods.rehearsal_training import rehearsal_training
from methods.ewc import ewc_training
from methods.lwf import lwf_training
from methods.bimeco import bimeco_training
from methods.lwf_with_bimeco import lwf_with_bimeco

import wandb
import pprint

def main(args):
    """
    In this function, we define the hyperparameters, instantiate the model, define the optimizer and loss function,
    and train the model.

    This function is going to be used to test methods about continual learning.

    :param args: arguments from the command line
    :return: None
    """
    wandb.login()

    parameters_dict = {}
    parameters_dict["exp_name"] = {"value": args.exp_name}
    parameters_dict["seed"] = {"value": args.seed}
    parameters_dict["epochs"] = {"value": args.epochs}
    parameters_dict["lr"] = {"value": args.lr}
    parameters_dict["lr_decay"] = {"value": args.lr_decay}
    parameters_dict["lr_patience"] = {"value": args.lr_patience}
    parameters_dict["lr_min"] = {"value": args.lr_min}
    parameters_dict["batch_size"] = {"value": args.batch_size}
    parameters_dict["num_tasks"] = {"value": args.num_tasks}

    parameters_dict["dataset"] = {"value": args.dataset}

    # parameters_dict["img_size"] = {"value": args.img_size}
    # parameters_dict["img_channels"] = {"value": args.img_channels}
    # parameters_dict["num_classes"] = {"value": args.num_classes}
    # parameters_dict["feature_dim"] = {"value": args.feature_dim}

    # EWC parameters
    # parameters_dict["ewc_lambda"] = {"value": args.ewc_lambda}

    # LwF parameters
    # parameters_dict["lwf_lambda"] = {"value": args.lwf_lambda}
    # parameters_dict["lwf_aux_lambda"] = {"value": args.lwf_aux_lambda}

    # BiMeCo parameters
    parameters_dict["memory_size"] = {"value": args.memory_size}
    # parameters_dict["bimeco_lambda_short"] = {"value": args.bimeco_lambda_short}
    # parameters_dict["bimeco_lambda_long"] = {"value": args.bimeco_lambda_long}
    # parameters_dict["bimeco_lambda_diff"] = {"value": args.bimeco_lambda_diff}
    # parameters_dict["m"] = {"value": args.m}

    # ================================================================================================================
    # Training with LwF
    # ================================================================================================================
    parameters_dict.update({
        "lwf_lambda": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5
        },
        "lwf_aux_lambda": {
            "distribution": "uniform",
            "min": 0.25,
            "max": 0.85
        }
    })

    # ================================================================================================================
    # Training with BiMeCo
    # ================================================================================================================
    parameters_dict.update({
        "bimeco_lambda_short": {
            "distribution": "uniform",
            "min": 1,
            "max": 20
        },
        "bimeco_lambda_long": {
            "distribution": "uniform",
            "min": 1,
            "max": 20
        },  
        "bimeco_lambda_diff": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 6
        },
        "m": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.99
        }
    })

    # ================================================================================================================

    sweep_config = {
        "name": args.exp_name,
        "method": "random",
        "metric": {
            "name": "Task2/test_avg_accuracy",
            "goal": "maximize"
        },
        "parameters": parameters_dict
    }
    pprint.pprint(sweep_config)

    # ================================================================================================================

    # Set the seed
    torch.manual_seed(args.seed)
    
    # Determine the operating system
    system_platform = platform.system()

    # Create the folders to save the models
    models_saved_path = f'./models/models_saved/{args.exp_name}'
    if os.path.exists(models_saved_path):
        if system_platform == 'Windows':
            # Use shutil.rmtree for Windows
            shutil.rmtree(models_saved_path)
        else:
            # Use os.system('rm -rf') for Unix-like systems
            os.system(f'rm -rf {models_saved_path}')
    os.makedirs(models_saved_path, exist_ok=True)

    # Create the folders to save the results
    results_path = f'./results/{args.exp_name}'
    if os.path.exists(results_path):
        if system_platform == 'Windows':
            # Use shutil.rmtree for Windows
            shutil.rmtree(results_path)
        else:
            # Use os.system('rm -rf') for Unix-like systems
            os.system(f'rm -rf {results_path}')
    os.makedirs(results_path, exist_ok=True)

    # Get the datasets
    if args.dataset == "mnist":
        datasets = get_dataset_mnist(args)
    elif args.dataset == "cifar10":
        datasets = get_dataset_cifar10(args)
    elif args.dataset == "cifar100":
        datasets = get_dataset_cifar100(args)
    elif args.dataset == "cifar100_alternative_dist":
        datasets = get_dataset_cifar100_alternative_dist(args)
        
    # ================================================================================================================
    # Project: "LwF"
    sweep_id = wandb.sweep(sweep_config, project="LwF")

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            lwf_training(datasets, args, config)
    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()

    # # Project: "LwF with criterion ANCL"
    sweep_id = wandb.sweep(sweep_config, project="LwF with criterion ANCL")

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            lwf_training(datasets, args, config, aux_training=False, criterion_bool=True)
    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()

    # # Project: "LwF with Auxiliar Network"
    sweep_id = wandb.sweep(sweep_config, project="LwF with Auxiliar Network")

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            lwf_training(datasets, args, config, aux_training=True)
    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()

    # # Project: "LwF with Auxiliar Network and criterion ANCL"
    sweep_id = wandb.sweep(sweep_config, project="LwF with Auxiliar Network and criterion ANCL")

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            lwf_training(datasets, args, config, aux_training=True, criterion_bool=True)
    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()
    # ================================================================================================================


    # ================================================================================================================
    # Project: "BiMeCo"
    sweep_id = wandb.sweep(sweep_config, project="BiMeCo")

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            bimeco_training(datasets, args, config)

    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()
    # ================================================================================================================


    # ================================================================================================================
    # Project: "LwF + BiMeCo"
    sweep_id = wandb.sweep(sweep_config, project="LwF + BiMeCo")

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            lwf_with_bimeco(datasets, args, config)

    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()

    # Project: "LwF + BiMeCo with criterion ANCL"
    sweep_id = wandb.sweep(sweep_config, project="LwF + BiMeCo with criterion ANCL")
    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            lwf_with_bimeco(datasets, args, config, aux_training=False, criterion_bool=True)

    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()

    # Project: "LwF + BiMeCo with auxiliar network"
    sweep_id = wandb.sweep(sweep_config, project="LwF + BiMeCo with auxiliar network")
    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            lwf_with_bimeco(datasets, args, config, aux_training=True, criterion_bool=False)

    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()

    # Project: "LwF + BiMeCo with auxiliar network and criterion ANCL"
    sweep_id = wandb.sweep(sweep_config, project="LwF + BiMeCo with auxiliar network and criterion ANCL")
    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            lwf_with_bimeco(datasets, args, config, aux_training=True, criterion_bool=True)

    wandb.agent(sweep_id, train, count=args.num_swipes)
    wandb.finish()

    


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()

    # General parameters
    argparse.add_argument('--exp_name', type=str, default="test")
    argparse.add_argument('--seed', type=int, default=0)
    argparse.add_argument('--epochs', type=int, default=500) # 500
    argparse.add_argument('--lr', type=float, default=0.001) # 0.001
    argparse.add_argument('--lr_decay', type=float, default=5) # 5
    argparse.add_argument('--lr_patience', type=int, default=10) # 10
    argparse.add_argument('--lr_min', type=float, default=1e-7) # 1e-8
    argparse.add_argument('--batch_size', type=int, default=200) # 200
    argparse.add_argument('--num_tasks', type=int, default=2) # 2

    # Dataset parameters: mnist, cifar10, cifar100, cifar100_alternative_dist
    argparse.add_argument('--dataset', type=str, default="cifar100_alternative_dist")

    # EWC parameters
    argparse.add_argument('--ewc_lambda' , type=float, default=1000) # 1000

    # Distillation parameters (LwF)
    argparse.add_argument('--lwf_lambda' , type=float, default=0.80) # 1
    argparse.add_argument('--lwf_aux_lambda' , type=float, default=0.75) # 0.5

    # BiMeCo parameters
    argparse.add_argument('--memory_size' , type=int, default=22500)
    argparse.add_argument('--bimeco_lambda_short' , type=float, default=1.5)
    argparse.add_argument('--bimeco_lambda_long' , type=float, default=2.5)
    argparse.add_argument('--bimeco_lambda_diff' , type=float, default=4)
    argparse.add_argument('--m' , type=float, default=0.15) # Momentum

    # Sweep parameters
    argparse.add_argument('--num_swipes', type=int, default=25)

    # Run the main function
    main(argparse.parse_args())


    