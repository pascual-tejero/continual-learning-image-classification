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

from models.architectures.net_mnist import Net_mnist
from models.architectures.net_cifar10 import Net_cifar10
from models.architectures.net_cifar100 import Net_cifar100

from methods.ewc_class import EWC, normal_train, normal_val, ewc_train, ewc_validate, test

def ewc_training(datasets, args):
    
    """
    In this function, we train the model using the EWC approach.

    :param datasets: list of datasets
    :param args: arguments from the command line

    :return: test_acc_final: list with the test accuracy of each task and the test average accuracy

    """

    print("\n")
    print("="*100)
    print("Training on EWC approach...")
    print("="*100)

    path_file = f"./results/{args.exp_name}/EWC_{args.dataset}.xlsx" # Path to save the results
    workbook = xlsxwriter.Workbook(path_file) # Create the excel file
    test_acc_final = [] # List to save the test accuracy of each task and the test average accuracy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the excel file
    if args.dataset == "mnist":
        model = Net_mnist().to(device) # Instantiate the model
    elif args.dataset == "cifar10":
        model = Net_cifar10().to(device) # Instantiate the model
    elif args.dataset == "cifar100" or args.dataset == "cifar100_alternative_dist":
        model = Net_cifar100().to(device) # Instantiate the model

    for id_task, task in enumerate(datasets):
        print("="*100)
        print("="*100)

        patience = args.lr_patience # Patience for early stopping
        lr = args.lr # Learning rate
        best_val_loss = 1e20 # Validation loss of the previous epoch
        model_best = copy.deepcopy(model) # Save the best model so far

        optimizer = optim.Adam(model.parameters(), lr=args.lr) # Instantiate the optimizer     
        
        dicc_results = {"Train task":[], "Train epoch": [], "Train loss":[], "Val loss":[],
                         "Test task":[], "Test loss":[], "Test accuracy":[], "Test average accuracy": []}

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
                print("="*100)
                print(f"METHOD: EWC -> Train on task {id_task+1}, Epoch: {epoch+1}")

                # Training
                train_loss_epoch = normal_train(model, optimizer, train_loader)

                # Validation
                val_loss_epoch = normal_val(model, val_loader)

                # Test
                test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy = test(model, 
                                                                                         datasets, 
                                                                                         args) 

                # Append the results to dicc_results
                dicc_results = append_results(dicc_results, id_task+1, epoch+1, train_loss_epoch, 
                                            val_loss_epoch, test_tasks_id, test_tasks_loss, 
                                            test_tasks_accuracy, avg_accuracy)              

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
                            test_acc_final.append([test_tasks_accuracy, avg_accuracy]) # Append the test accuracy of each task and the test average accuracy
                            break
                        # reset patience and recover best model so far to continue training
                        patience = args.lr_patience
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        model.load_state_dict(model_best.state_dict())
                
                # Save the results of the epoch if it is the last epoch
                if epoch == args.epochs-1:
                    test_acc_final.append([test_tasks_accuracy, avg_accuracy]) # Append the test accuracy of each task and the test average accuracy
                
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience}")

                
        else:            
            # Load the previous trained model
            old_model = copy.deepcopy(model)

            # Load the previous model
            path_old_model = (f"./models/models_saved/{args.exp_name}/EWC_{args.dataset}/"
                              f"EWC_aftertask_{id_task}_{args.dataset}.pt")
            old_model.load_state_dict(torch.load(path_old_model))
                                                            
            for epoch in range(args.epochs):
                print("="*100)
                print(f"METHOD: EWC -> Train on task {id_task+1}, Epoch: {epoch+1}")

                # Training
                train_loss_epoch = ewc_train(model, optimizer, train_loader, 
                                             EWC(model, old_model, train_loader, args),
                                             importance=args.ewc_lambda)

                # Validation
                val_loss_epoch = ewc_validate(model, val_loader, 
                                              EWC(model, old_model, val_loader, args),
                                              importance=args.ewc_lambda)

                # Test
                test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy = test(model, 
                                                                                         datasets, 
                                                                                         args)

                # Append the results to dicc_results
                dicc_results = append_results(dicc_results, id_task+1, epoch+1, train_loss_epoch, 
                                            val_loss_epoch, test_tasks_id, test_tasks_loss, 
                                            test_tasks_accuracy, avg_accuracy)
                
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
                            # Append the test accuracy of each task and the test average accuracy
                            test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 
                            break
                        # reset patience and recover best model so far to continue training
                        patience = args.lr_patience
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        model_best = copy.deepcopy(model)
                        model.load_state_dict(model_best.state_dict())

                # Save the results of the epoch if it is the last epoch
                if epoch == args.epochs-1:
                    # Append the test accuracy of each task and the test average accuracy
                    test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 

                print(f"Learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience}")

        # Save the results (after each task)
        save_training_results(dicc_results, workbook, id_task+1, training_name="EWC")

        # Save the model
        save_model(model_best, args, id_task+1, method="EWC")

    # Close the excel file
    workbook.close()

    return test_acc_final



def append_results(dicc_results, id_task, epoch, train_loss_epoch, val_loss_epoch, 
                   test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy):

    # Append the results to dicc_results
    dicc_results["Train task"].append(id_task)
    dicc_results["Train epoch"].append(epoch)
    dicc_results["Train loss"].append(train_loss_epoch)
    dicc_results["Val loss"].append(val_loss_epoch)
    dicc_results["Test task"].append(test_tasks_id)
    dicc_results["Test loss"].append(test_tasks_loss)
    dicc_results["Test accuracy"].append(test_tasks_accuracy)
    dicc_results["Test average accuracy"].append(avg_accuracy)

    return dicc_results 
