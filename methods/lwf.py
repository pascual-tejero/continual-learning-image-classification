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

from methods.lwf_class import normal_train, normal_val, lwf_train, lwf_validate, test, lwf_train_aux, lwf_validate_aux
import wandb

def lwf_training(datasets, args, config, aux_training=False, criterion_bool=None):

    print("\n")
    print("="*100)
    print("Training on LwF approach...")
    print("="*100)

    if aux_training and not criterion_bool:
        # path_file = f'./results/{args.exp_name}/LwF_auxNetwork_{args.dataset}.xlsx'
        method_cl = "LwF_auxNetwork"
        method_print = "LwF with auxiliar network"
    elif not aux_training and criterion_bool:
        # path_file = f'./results/{args.exp_name}/LwF_criterionANCL_{args.dataset}.xlsx'
        method_cl = "LwF_criterionANCL"
        method_print = "LwF with criterion ANCL"
    elif aux_training and criterion_bool:
        # path_file = f'./results/{args.exp_name}/LwF_auxNetwork_criterionANCL_{args.dataset}.xlsx'
        method_cl = "LwF_auxNetwork_criterionANCL"
        method_print = "LwF with auxiliar network and criterion ANCL"
    else:
        # path_file = f'./results/{args.exp_name}/LwF_{args.dataset}.xlsx'
        method_cl = "LwF"
        method_print = "LwF"

    test_acc_final = []  # List to save the average accuracy of each task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the excel file
    if args.dataset == "mnist":
        model = Net_mnist().to(device)  # Instantiate the model
    elif args.dataset == "cifar10":
        model = Net_cifar10().to(device)  # Instantiate the model
    elif args.dataset == "cifar100" or args.dataset == "cifar100_alternative_dist":
        model = Net_cifar100().to(device)  # Instantiate the model

    for id_task, task in enumerate(datasets):
        print("="*100)
        print("="*100)
        
        patience = args.lr_patience # Patience for early stopping
        lr = args.lr # Learning rate
        best_val_loss = 1e20 # Validation loss of the previous epoch
        model_best = copy.deepcopy(model) # Save the best model so far

        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Instantiate the optimizer

        dicc_results = {"Train task": [], "Train epoch": [], "Train loss": [], "Val loss": [],
                        "Test task": [], "Test loss": [], "Test accuracy": [], "Test average accuracy": []}

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
                print("="*100)
                print(f"METHOD: {method_print} -> Train on task {id_task+1}, Epoch: {epoch+1}")

                # Training
                train_loss_epoch = normal_train(model, optimizer, train_loader, criterion_bool)

                # Validation
                val_loss_epoch = normal_val(model, val_loader, criterion_bool)

                # Test
                test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy = test(model, 
                                                                                         datasets, 
                                                                                         args)

                # Append the results to dicc_results
                wandb.log({"Task 1/Epoch": epoch+1, 
                           "Task 1/Train loss": train_loss_epoch, 
                           "Task 1/Validation loss": val_loss_epoch, 
                           "Task 1/Test loss task 1": test_tasks_loss[0],
                           "Task 1/Test loss task 2": test_tasks_loss[1], 
                           "Task 1/Test accuracy task 1": test_tasks_accuracy[0],
                           "Task 1/Test accuracy task 2": test_tasks_accuracy[1],
                           "Task 1/Test average accuracy": avg_accuracy})
                
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
                            test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 
                            break
                        # reset patience and recover best model so far to continue training
                        patience = args.lr_patience
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        model.load_state_dict(model_best.state_dict())

                print(f"Current learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience}")

                # Save the results of the epoch if it is the last epoch
                if epoch == args.epochs-1:
                    test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 
                
            
        else:
            
            if aux_training:

                patience_aux = args.lr_patience # Patience for early stopping
                lr_aux = args.lr # Learning rate
                best_val_loss_aux = 1e20 # Validation loss of the previous epoch
                auxiliar_network = copy.deepcopy(model_best).to(device)
                optimizer_aux = optim.Adam(auxiliar_network.parameters(), lr=args.lr)  # Instantiate the optimizer

                for epoch in range(args.epochs):
                    print("="*100)
                    print("Train the auxiliar network...")
                    print(f"METHOD: {method_print} -> Train on task {id_task+1}, Epoch: {epoch+1}")

                    normal_train(auxiliar_network, optimizer_aux, train_loader, criterion_bool)
                    val_loss_epoch_aux = normal_val(auxiliar_network, val_loader, criterion_bool)
                    test(auxiliar_network, datasets, args)

                    # Early stopping
                    if val_loss_epoch_aux < best_val_loss_aux:
                        best_val_loss_aux = val_loss_epoch_aux
                        patience_aux = args.lr_patience
                        model_best_aux = copy.deepcopy(auxiliar_network)
                    else:
                        # if the loss does not go down, decrease patience
                        patience_aux -= 1
                        if patience_aux <= 0:
                            # if it runs out of patience, reduce the learning rate
                            lr_aux /= args.lr_decay
                            print(' lr={:.1e}'.format(lr), end='')
                            if lr_aux < args.lr_min:
                                # if the lr decreases below minimum, stop the training session
                                print()
                                # test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 
                                break
                            # reset patience and recover best model so far to continue training
                            patience_aux = args.lr_patience
                            for param_group in optimizer_aux.param_groups:
                                param_group['lr'] = lr_aux
                            auxiliar_network.load_state_dict(model_best_aux.state_dict())
                    
                    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience_aux}")

                    if epoch == args.epochs-1:
                        auxiliar_network = copy.deepcopy(model_best_aux).to(device)
                        
                torch.save(auxiliar_network.state_dict(), (f"./models/models_saved/{args.exp_name}/{method_cl}_{args.dataset}/"
                                                           f"Aux_Network_task_{id_task+1}_{args.dataset}.pt"))
            
                auxiliar_network.eval()
                for param in auxiliar_network.parameters():
                    param.requires_grad = False
                
            # Load the previous model
            old_model = copy.deepcopy(model).to(device)

            path_old_model = (f"./models/models_saved/{args.exp_name}/{method_cl}_{args.dataset}/"
                            f"{method_cl}_aftertask_{id_task}_{args.dataset}.pt")
            old_model.load_state_dict(torch.load(path_old_model))
            old_model.eval()

            for epoch in range(args.epochs):
                print("="*100)
                print(f"METHOD: {method_print} -> Train on task {id_task+1}, Epoch: {epoch+1}")

                if not aux_training:
                    # Training
                    train_loss_epoch = lwf_train(model, old_model, optimizer, train_loader, 
                                                 config["lwf_lambda"], criterion_bool)

                    # Validation
                    val_loss_epoch = lwf_validate(model, old_model, val_loader, config["lwf_lambda"], criterion_bool)
                
                else:
                    # Training
                    train_loss_epoch = lwf_train_aux(model, old_model, optimizer, train_loader, 
                                                 config["lwf_lambda"], auxiliar_network, config["lwf_aux_lambda"],
                                                 criterion_bool)

                    # Validation
                    val_loss_epoch = lwf_validate_aux(model, old_model, val_loader, config["lwf_lambda"],
                                                  auxiliar_network, args.lwf_aux_lambda, criterion_bool)

                # Test
                test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy = test(model, 
                                                                                         datasets, 
                                                                                         args)

                # Append the results to dicc_results
                wandb.log({f"Task {id_task+1}/Epoch": epoch+1, 
                           f"Task {id_task+1}/Train loss": train_loss_epoch, 
                           f"Task {id_task+1}/Validation loss": val_loss_epoch, 
                           f"Task {id_task+1}/Test loss task 1": test_tasks_loss[0],
                           f"Task {id_task+1}/Test loss task 2": test_tasks_loss[1], 
                           f"Task {id_task+1}/Test accuracy task 1": test_tasks_accuracy[0],
                           f"Task {id_task+1}/Test accuracy task 2": test_tasks_accuracy[1],
                           f"Task {id_task+1}/Test average accuracy": avg_accuracy})

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
                            test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 
                            break
                        # reset patience and recover best model so far to continue training
                        patience = args.lr_patience
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        model.load_state_dict(model_best.state_dict())

                print(f"Learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience}")

                # Save the results of the epoch if it is the last epoch 
                if epoch == args.epochs-1:
                    test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 
                
        # Save the model
        save_model(model_best, args, id_task+1, method=method_cl)

    return test_acc_final


