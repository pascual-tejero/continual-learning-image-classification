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
from methods.lwf_with_membuffer import lwf_with_membuffer


def main(args):
    """
    In this function, we define the hyperparameters, instantiate the model, define the optimizer and loss function,
    and train the model.

    This function is going to be used to test methods about continual learning.

    :param args: arguments from the command line
    :return: None
    """
    print("Arguments: ", args)
    
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
    elif args.dataset == "cifar100-alternative-dist":
        datasets = get_dataset_cifar100_alternative_dist(args)

    # # Create a dictionary to save the results
    dicc_results_test = {}

    # Train the model using the naive approach (no continual learning) for fine-tuning
    dicc_results_test["Fine-tuning"] = naive_training(datasets, args)

    # Train the model using the naive approach (no continual learning) for joint training
    dicc_results_test["Joint datasets"] = naive_training(datasets, args, joint_datasets=True)

    # # Train the model using the rehearsal approach
    dicc_results_test["Rehearsal 10%"] = rehearsal_training(datasets, args, rehearsal_prop=0.1, random_rehearsal=True)
    dicc_results_test["Rehearsal 30%"] = rehearsal_training(datasets, args, rehearsal_prop=0.3, random_rehearsal=True)
    dicc_results_test["Rehearsal 50%"] = rehearsal_training(datasets, args, rehearsal_prop=0.5, random_rehearsal=True)

    # # Train the model using the EWC approach
    dicc_results_test["EWC"] = ewc_training(datasets, args)

    # Train the model using the LwF approach
    dicc_results_test["LwF"] = lwf_training(datasets, args)
    dicc_results_test["LwF lossANCL"] = lwf_training(datasets, args, aux_training=False, loss_ANCL=True)

    dicc_results_test["LwF AuxNet"] = lwf_training(datasets, args, aux_training=True)
    dicc_results_test["LwF AuxNet lossANCL"] = lwf_training(datasets, args, aux_training=True, loss_ANCL=True)

    # # Train the model using the BiMeCo approach
    dicc_results_test["BiMeCo"] = bimeco_training(datasets, args)

    dicc_results_test["LwF + BiMeCo"] = lwf_with_bimeco(datasets, args)
    dicc_results_test["LwF lossANCL + BiMeCo "] = lwf_with_bimeco(datasets, args, aux_training=False, loss_ANCL=True)
    dicc_results_test["LwF AuxNet + BiMeCo"] = lwf_with_bimeco(datasets, args, aux_training=True)
    dicc_results_test["LwF AuxNet lossANCL + BiMeCo "] = lwf_with_bimeco(datasets, args, aux_training=True, loss_ANCL=True)

    # Save the results
    save_global_results(dicc_results_test, args)

    # Create the .txt file and save the arguments
    with open(f'./results/{args.exp_name}/args_{args.exp_name}_{args.dataset}.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key} : {value}\n')


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()

    # General parameters
    argparse.add_argument('--exp_name', type=str, default="CL_methods")
    argparse.add_argument('--seed', type=int, default=0)
    argparse.add_argument('--epochs', type=int, default=1) # 500
    argparse.add_argument('--lr', type=float, default=0.001) # 0.001
    argparse.add_argument('--lr_decay', type=float, default=5) # 5
    argparse.add_argument('--lr_patience', type=int, default=10) # 10
    argparse.add_argument('--lr_min', type=float, default=1e-8) # 1e-8
    argparse.add_argument('--batch_size', type=int, default=200) # 200
    argparse.add_argument('--num_tasks', type=int, default=2) # 2

    # Dataset parameters: mnist, cifar10, cifar100, cifar100-alternative-dist
    argparse.add_argument('--dataset', type=str, default="cifar100")

    # EWC parameters
    argparse.add_argument('--ewc_lambda' , type=float, default=100000) # 1000

    # Distillation parameters (LwF)
    argparse.add_argument('--lwf_lambda' , type=float, default=0.8) # 1
    argparse.add_argument('--lwf_aux_lambda' , type=float, default=0.75) # 0.5

    # BiMeCo parameters
    argparse.add_argument('--memory_size' , type=int, default=22500)
    argparse.add_argument('--bimeco_lambda_short' , type=float, default=1.5)
    argparse.add_argument('--bimeco_lambda_long' , type=float, default=2.5)
    argparse.add_argument('--bimeco_lambda_diff' , type=float, default=4)
    argparse.add_argument('--m' , type=float, default=0.15) # Momentum

    # Run the main function
    main(argparse.parse_args())
    
