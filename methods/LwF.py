import torch
import torch.nn.functional as F
import torch.optim as optim

import xlsxwriter
import os
import sys
import copy

sys.path.append('../')

from utils.save_training_results import save_training_results
from utils.utils import save_model

from models.net_mnist import Net_mnist
from models.net_cifar10 import Net_cifar10
from models.net_cifar100 import Net_cifar100

from methods.lwf_class import normal_train, normal_val, lwf_train, lwf_validate, test

def lwf_training(datasets, args):

    print("------------------------------------------")
    print("Training on LwF approach...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the excel file
    if args.dataset == "mnist":
        model = Net_mnist().to(device)  # Instantiate the model

    elif args.dataset == "cifar10":
        model = Net_cifar10().to(device)  # Instantiate the model

    elif args.dataset == "cifar100":
        model = Net_cifar100().to(device)  # Instantiate the model

    path_file = f'./results/{args.exp_name}/results_lwf_{args.dataset}.xlsx'

    workbook = xlsxwriter.Workbook(path_file)  # Create the excel file

    avg_acc_list = []  # List to save the average accuracy of each task

    for id_task, task in enumerate(datasets):

        patience = args.lr_patience # Patience for early stopping
        lr = args.lr # Learning rate
        best_val_loss = 1e20 # Validation loss of the previous epoch
        model_best = copy.deepcopy(model) # Save the best model so far

        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Instantiate the optimizer

        dicc_results = {"Train task": [], "Train epoch": [], "Train loss": [], "Val loss": [],
                        "Test task": [], "Test loss": [], "Test accuracy": [], "Test average accuracy": []}
        print("------------------------------------------")

        train_dataset, val_dataset, _ = task  # Get the images and labels from the task

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
                print(f"Task {id_task + 1} -> Epoch: {epoch + 1}")

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
                
                # Save the results of the epoch
                if epoch == args.epochs-1:
                    avg_acc_list.append(avg_acc)

                # Early stopping
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    patience = args.lr_patience
                    model_best = copy.deepcopy(model)
                else:
                    # if the loss does not go down, decrease patience
                    patience -= 1
                    if patience <= 0:
                        # if it runs out of patience, reduce the learning rate
                        lr /= args.lr_decay
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < args.lr_min:
                            # if the lr decreases below minimum, stop the training session
                            print()
                            avg_acc_list.append(avg_acc) # Append the average accuracy of the task
                            break
                        # reset patience and recover best model so far to continue training
                        patience = args.lr_patience
                        optimizer.param_groups[0]['lr'] = lr
                        model.load_state_dict(model_best.state_dict())
                
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience}")
            
        else:
            
            # Load the previous model
            if args.dataset == "mnist":
                old_model = Net_mnist().to(device)
            elif args.dataset == "cifar10":
                old_model = Net_cifar10().to(device)
            elif args.dataset == "cifar100":
                old_model = Net_cifar100().to(device)

            # Load the previous model
            path_old_model = (f"./models/models_saved/{args.exp_name}/lwf_training_{args.dataset}/"
                               f"model_lwf_aftertask_{id_task}_{args.dataset}.pt")
            old_model.load_state_dict(torch.load(path_old_model))

            for epoch in range(args.epochs):
                print("------------------------------------------")
                print(f"Task {id_task + 1} -> Epoch: {epoch + 1}")

                # Training
                train_loss_epoch = lwf_train(model, 
                                             old_model,
                                             optimizer, 
                                             train_loader, 
                                             alpha=args.lwf_lambda)

                # Validation
                val_loss_epoch = lwf_validate(model, 
                                              old_model,
                                              val_loader, 
                                              alpha=args.lwf_lambda)

                # Test
                test_task_list, test_loss_list, test_acc_list, avg_acc = test(model, datasets, args)

                # Append the results to dicc_results
                dicc_results, avg_acc_list = append_results(dicc_results, avg_acc_list, id_task, epoch,
                                                            train_loss_epoch, val_loss_epoch, test_task_list,
                                                            test_loss_list, test_acc_list, avg_acc, args)

                # Save the results of the epoch
                if epoch == args.epochs-1:
                    avg_acc_list.append(avg_acc)

                # Early stopping
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    patience = args.lr_patience
                    model_best = copy.deepcopy(model)
                else:
                    # if the loss does not go down, decrease patience
                    patience -= 1
                    if patience <= 0:
                        # if it runs out of patience, reduce the learning rate
                        lr /= args.lr_decay
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < args.lr_min:
                            # if the lr decreases below minimum, stop the training session
                            print()
                            avg_acc_list.append(avg_acc) # Append the average accuracy of the task
                            break
                        # reset patience and recover best model so far to continue training
                        patience = args.lr_patience
                        optimizer.param_groups[0]['lr'] = lr
                        model.load_state_dict(model_best.state_dict())
                
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience}")

        # Save the results (after each task)
        save_training_results(dicc_results, workbook, task=id_task, training_name="LwF")

        # Save the model
        save_model(model_best, args, id_task_dataset=id_task+1, task="lwf")

    # Close the excel file
    workbook.close()

    return avg_acc_list

def append_results(dicc_results, avg_acc_list, id_task, epoch, train_loss_epoch, val_loss_epoch, test_task_list,
                   test_loss_list, test_acc_list, avg_acc, args):

    # Append the results to dicc_results
    dicc_results["Train task"].append(id_task + 1)
    dicc_results["Train epoch"].append(epoch + 1)
    dicc_results["Train loss"].append(train_loss_epoch)
    dicc_results["Val loss"].append(val_loss_epoch)
    dicc_results["Test task"].append(test_task_list)
    dicc_results["Test loss"].append(test_loss_list)
    dicc_results["Test accuracy"].append(test_acc_list)
    dicc_results["Test average accuracy"].append(avg_acc)

    return dicc_results, avg_acc_list
