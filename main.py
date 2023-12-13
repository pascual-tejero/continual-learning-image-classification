import torch

import argparse
import os 

from utils.get_dataset_mnist import get_dataset_mnist
from utils.get_dataset_cifar10 import get_dataset_cifar10
from utils.get_dataset_cifar100 import get_dataset_cifar100
from utils.save_global_results import save_global_results

from methods.naive_training import naive_training
from methods.rehearsal_training import rehearsal_training
from methods.ewc import ewc_training
from methods.lwf import lwf_training


def main(args):
    """
    In this function, we define the hyperparameters, instantiate the model, define the optimizer and loss function,
    and train the model.

    This function is going to be used to test methods about continual learning.

    :param args: arguments from the command line
    :return: None
    """
    # Set the seed
    torch.manual_seed(args.seed)

    # Create a dictionary to save the results
    dicc_results_test = {}
    
    # Create the folders to save the models
    if os.path.exists(f'./models/models_saved/{args.exp_name}'):
        os.system(f'rm -rf ./models/models_saved/{args.exp_name}')
    os.makedirs(f'./models/models_saved/{args.exp_name}', exist_ok=True)

    # Create the folders to save the results
    if os.path.exists(f'./results/{args.exp_name}'):
        os.system(f'rm -rf ./results/{args.exp_name}')
    os.makedirs(f'./results/{args.exp_name}', exist_ok=True)

    if args.dataset == "mnist":
        datasets = get_dataset_mnist(args)
    elif args.dataset == "cifar10":
        datasets = get_dataset_cifar10(args)
    elif args.dataset == "cifar100":
        datasets = get_dataset_cifar100(args)
        
    # Train the model using the naive approach (no continual learning) for fine-tuning
    dicc_results_test["Fine-tuning"] = naive_training(datasets, args)

    # Train the model using the naive approach (no continual learning) for joint training
    dicc_results_test["Joint datasets"] = naive_training(datasets, args, joint_datasets=True)

    # Train the model using the rehearsal approach
    dicc_results_test["Rehearsal 0.1"] = rehearsal_training(datasets, args, rehearsal_percentage=0.1, random_rehearsal=True)
    dicc_results_test["Rehearsal 0.3"] = rehearsal_training(datasets, args, rehearsal_percentage=0.3, random_rehearsal=True)
    dicc_results_test["Rehearsal 0.5"] = rehearsal_training(datasets, args, rehearsal_percentage=0.5, random_rehearsal=True)

    # Train the model using the EWC approach
    dicc_results_test["EWC"] = ewc_training(datasets, args)

    # Train the model using the LwF approach
    dicc_results_test["LwF"] = lwf_training(datasets, args)
    
    # Save the results
    save_global_results(dicc_results_test, args)

    # Create the .txt file and save the arguments
    with open(f'./results/{args.exp_name}/args_{args.exp_name}_{args.dataset}.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key} : {value}\n')


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()

    # General parameters
    argparse.add_argument('--exp_name', type=str, default="exp1")
    argparse.add_argument('--seed', type=int, default=0)
    argparse.add_argument('--epochs', type=int, default=1)
    argparse.add_argument('--lr', type=float, default=0.001)
    argparse.add_argument('--lr_decay', type=float, default=5)
    argparse.add_argument('--lr_patience', type=int, default=5)
    argparse.add_argument('--lr_min', type=float, default=1e-6)
    argparse.add_argument('--batch_size', type=int, default=200)
    argparse.add_argument('--num_tasks', type=int, default=2)
    # argparse.add_argument('--scheduler_step_size', type=int, default=7)
    # argparse.add_argument('--scheduler_gamma', type=float, default=0.3)

    # Dataset parameters: mnist, cifar10, cifar100
    argparse.add_argument('--dataset', type=str, default="cifar10")

    # EWC parameters
    argparse.add_argument('--ewc_lambda' , type=float, default=1000)

    # Distillation parameters
    argparse.add_argument('--lwf_lambda' , type=float, default=1)


    # Run the main function
    main(argparse.parse_args())