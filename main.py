import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import argparse
import xlsxwriter
import os 
import random

from utils.get_datasets import get_datasets
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
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create a dictionary to save the results
    dicc_avg_acc = {}
    
    # Get the dataloader
    datasets = get_datasets(args)
    quit()
    # Train the model using the naive approach
    dicc_avg_acc["Naive"] = naive_training(datasets, args)

    # # Train the model using the rehearsal approach
    dicc_avg_acc["Rehearsal 0.1"] = rehearsal_training(datasets, args, rehearsal_percentage=0.1)
    dicc_avg_acc["Rehearsal 0.3"] = rehearsal_training(datasets, args, rehearsal_percentage=0.3)
    dicc_avg_acc["Rehearsal 0.5"] = rehearsal_training(datasets, args, rehearsal_percentage=0.5)

    # # Train the model using the EWC approach
    dicc_avg_acc["EWC"] = ewc_training(datasets, args)

    # Train the model using the LwF approach
    dicc_avg_acc["LwF"] = lwf_training(datasets, args)

    # Save the results
    save_global_results(dicc_avg_acc, args)
    

    

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()

    # General parameters
    argparse.add_argument('--batch_size', type=int, default=50)
    argparse.add_argument('--epochs', type=int, default=20)
    argparse.add_argument('--lr', type=float, default=0.001)
    argparse.add_argument('--dataset', type=str, default="mnist")
    # argparse.add_argument('--dataset', type=str, default="cifar10")

    # EWC parameters
    argparse.add_argument('--ewc_lambda' , type=float, default=5)

    # Distillation parameters
    argparse.add_argument('--lwf_lambda' , type=float, default=5)

    # Run the main function
    main(argparse.parse_args())