import torch

import argparse
import os 



from utils.get_dataset_mnist import get_dataset_mnist
from utils.get_dataset_cifar10 import get_dataset_cifar10
from utils.get_dataset_cifar100 import get_dataset_cifar100
from utils.save_global_results import save_global_results

from methods.naive_training import naive_training
from methods.rehearsal_training import rehearsal_training
from methods.EWC import ewc_training
from methods.LwF import lwf_training


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
    dicc_avg_acc = {}
    
    # # Get the dataloader
    # if args.dataset == "mnist":
    #     if os.path.exists('./results/mnist/') == False:
    #         os.makedirs('./results/mnist/')
    #     datasets = get_dataset_mnist(args)
    # elif args.dataset == "cifar10":
    #     if os.path.exists('./results/cifar10/') == False:
    #         os.makedirs('./results/cifar10/')
    #     datasets = get_dataset_cifar10(args)
    # elif args.dataset == "cifar100":
    #     if os.path.exists('./results/cifar100/') == False:
    #         os.makedirs('./results/cifar100/')
    #     datasets = get_dataset_cifar100(args)
    
        # Get the dataloader
    if args.dataset == "mnist":
        if os.path.exists('./results/mnist_test/') == False:
            os.makedirs('./results/mnist_test/')
        datasets = get_dataset_mnist(args)
    elif args.dataset == "cifar10":
        if os.path.exists('./results/cifar10_test/') == False:
            os.makedirs('./results/cifar10_test/')
        datasets = get_dataset_cifar10(args)
    elif args.dataset == "cifar100":
        if os.path.exists('./results/cifar100_test/') == False:
            os.makedirs('./results/cifar100_test/')
        datasets = get_dataset_cifar100(args)
    
    
    # Train the model using the naive approach (no continual learning) for fine-tuning
    dicc_avg_acc["Finetuning"] = naive_training(datasets, args)

    # Train the model using the naive approach (no continual learning) for joint training
    dicc_avg_acc["Joint training"] = naive_training(datasets, args, joint_training=True)

    # Train the model using the rehearsal approach
    dicc_avg_acc["Rehearsal 0.1"] = rehearsal_training(datasets, args, rehearsal_percentage=0.1, random_rehearsal=True)
    dicc_avg_acc["Rehearsal 0.3"] = rehearsal_training(datasets, args, rehearsal_percentage=0.3, random_rehearsal=True)
    dicc_avg_acc["Rehearsal 0.5"] = rehearsal_training(datasets, args, rehearsal_percentage=0.5, random_rehearsal=True)

    # # Train the model using the EWC approach
    dicc_avg_acc["EWC"] = ewc_training(datasets, args)

    # # Train the model using the LwF approach
    dicc_avg_acc["LwF"] = lwf_training(datasets, args)

    # Save the results
    save_global_results(dicc_avg_acc, args)
    

    

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()

    # General parameters
    argparse.add_argument('--seed', type=int, default=0)
    argparse.add_argument('--epochs', type=int, default=2)
    argparse.add_argument('--lr', type=float, default=0.001)
    argparse.add_argument('--batch_size', type=int, default=20)
    argparse.add_argument('--num_tasks', type=int, default=4)
    argparse.add_argument('--scheduler_step_size', type=int, default=0.01)
    argparse.add_argument('--scheduler_gamma', type=float, default=0.1)

    # Dataset parameters: mnist, cifar10, cifar100
    argparse.add_argument('--dataset', type=str, default="cifar10")

    # EWC parameters
    argparse.add_argument('--ewc_lambda' , type=float, default=5)

    # Distillation parameters
    argparse.add_argument('--lwf_lambda' , type=float, default=10)

    # Run the main function
    main(argparse.parse_args())