import torch
import torch.nn.functional as F
import torch.optim as optim

import xlsxwriter
import os
import sys

sys.path.append('../')

from utils.save_training_results import save_training_results

from models.net_mnist import Net_mnist
from models.net_cifar10 import Net_cifar10
from models.net_cifar100 import Net_cifar100

from methods.ewc_class import EWC, normal_train, normal_val, ewc_train, ewc_validate, test

def ewc_training(datasets, args):
    
    """
    In this function, we train the model using the naive approach, which is training the model on the first dataset
    and then training the model on the second dataset.
    """

    print("------------------------------------------")
    print("Training on EWC approach...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the excel file
    if args.dataset == "mnist":
        path_file = "./results/mnist/results_mnist_ewc.xlsx"
        model = Net_mnist().to(device) # Instantiate the model

    elif args.dataset == "cifar10":
        path_file = "./results/cifar10/results_cifar10_ewc.xlsx"
        model = Net_cifar10().to(device) # Instantiate the model

    elif args.dataset == "cifar100":
        path_file = "./results/cifar100/results_cifar100_ewc.xlsx"
        model = Net_cifar100().to(device) # Instantiate the model
    
    if os.path.exists(path_file): # If the file exists
        os.remove(path_file) # Remove the file if it exists
    workbook = xlsxwriter.Workbook(path_file) # Create the excel file

    avg_acc_list = [] # List to save the average accuracy of each task

    # EWC_obj = EWC(model, datasets) # Instantiate the EWC class

    for id_task, task in enumerate(datasets):

        optimizer = optim.Adam(model.parameters(), lr=args.lr) # Instantiate the optimizer     

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, 
                                                gamma=args.scheduler_gamma)  
        
        dicc_results = {"Train task":[], "Train epoch": [], "Train loss":[], "Val loss":[],
                         "Test task":[], "Test loss":[], "Test accuracy":[], "Test average accuracy": []}
        print("------------------------------------------")

        train_dataset, val_dataset, _ = task # Get the images and labels from the task
        
        # Make the dataloader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
        
        if id_task == 0:

            for epoch in range(args.epochs):
                print("------------------------------------------")
                print(f"Task {id_task+1} -> Epoch: {epoch+1}, Learning rate: {scheduler.get_last_lr()[0]}")

                # Training
                train_loss_epoch = normal_train(model, optimizer, train_loader)

                # Validation
                val_loss_epoch = normal_val(model, val_loader)

                # Test
                test_task_list, test_loss_list, test_acc_list, avg_acc = test(model, datasets, args) 

                # Append the results to dicc_results
                dicc_results, avg_acc_list = append_results(dicc_results, avg_acc_list, id_task, epoch, 
                                                            train_loss_epoch, val_loss_epoch, test_task_list,
                                                            test_loss_list, test_acc_list, avg_acc, args) 
                scheduler.step() # Update the learning rate

        else:
            old_tasks_train = []
            old_tasks_val = []

            for i in range(id_task):
                old_tasks_train.append(datasets[i][0]) # Get the images and labels from the task
                old_tasks_val.append(datasets[i][1]) # Get the images and labels from the task

            # Get a random sample from the old tasks
            old_tasks_train = torch.utils.data.ConcatDataset(old_tasks_train)
            old_tasks_val = torch.utils.data.ConcatDataset(old_tasks_val)

            # Make the dataloader
            old_tasks_train_loader = torch.utils.data.DataLoader(dataset=old_tasks_train,
                                                                batch_size=args.batch_size,
                                                                shuffle=True)
            old_tasks_val_loader = torch.utils.data.DataLoader(dataset=old_tasks_val,
                                                                batch_size=args.batch_size,
                                                                shuffle=True)
                                                            
            for epoch in range(args.epochs):
                print("------------------------------------------")
                print(f"Task {id_task+1} -> Epoch: {epoch+1}, Learning rate: {scheduler.get_last_lr()[0]}")

                # Training
                train_loss_epoch = ewc_train(model, optimizer, train_loader, EWC(model, old_tasks_train_loader),
                                            importance=args.ewc_lambda)

                # Validation
                val_loss_epoch = ewc_validate(model, val_loader, EWC(model, old_tasks_val_loader),
                                            importance=args.ewc_lambda)
                
                # Test
                test_task_list, test_loss_list, test_acc_list, avg_acc = test(model, datasets, args)

                # Append the results to dicc_results
                dicc_results, avg_acc_list = append_results(dicc_results, avg_acc_list, id_task, epoch, 
                                                            train_loss_epoch, val_loss_epoch, test_task_list,
                                                            test_loss_list, test_acc_list, avg_acc, args) 

                scheduler.step() # Update the learning rate               

        # Save the results (after each task)
        save_training_results(dicc_results, workbook, task=id_task, training_name="EWC")

    # Close the excel file
    workbook.close()

    return avg_acc_list

def append_results(dicc_results, avg_acc_list, id_task, epoch, train_loss_epoch, val_loss_epoch, test_task_list,
                    test_loss_list, test_acc_list, avg_acc, args):

    # Append the results to dicc_results
    dicc_results["Train task"].append(id_task+1)
    dicc_results["Train epoch"].append(epoch+1)
    dicc_results["Train loss"].append(train_loss_epoch)
    dicc_results["Val loss"].append(val_loss_epoch)
    dicc_results["Test task"].append(test_task_list)
    dicc_results["Test loss"].append(test_loss_list)
    dicc_results["Test accuracy"].append(test_acc_list)
    dicc_results["Test average accuracy"].append(avg_acc)

    if epoch == args.epochs-1:
        avg_acc_list.append(avg_acc) 

    return dicc_results, avg_acc_list
