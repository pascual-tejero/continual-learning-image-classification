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

def naive_training(datasets, args, joint_datasets=False):
    
    """
    In this function, we train the model using the naive approach (no continual learning).

    :param datasets: list of datasets
    :param args: arguments from the command line
    :param joint_datasets: boolean to indicate if we are training with joint datasets or not
    
    :return: test_acc_final: list to save the test accuracy of each task and the test average accuracy
    """
    print("\n")
    print("="*100)
    if not joint_datasets:
        print("Training: NAIVE approach -> FINE-TUNING...")
    else:
        print("Training: NAIVE approach -> JOINT-DATASETS...")
    print("="*100)

    # Create the excel file
    if joint_datasets:
        path_file = f"./results/{args.exp_name}/naive_joint-training_{args.dataset}.xlsx"

        concat_datasets = [] # List to save the joint dataset
        train_dataset, val_dataset, test_dataset = datasets[0] # Get the images and labels from the task

        if len(datasets) > 1: # If there are more than one task
            for i in range(1,len(datasets)): # For each task
                train_dataset_i, val_dataset_i, test_dataset_i = datasets[i] # Get the images and labels from the task
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_i]) # Concatenate the datasets
                val_dataset = torch.utils.data.ConcatDataset([val_dataset, val_dataset_i]) # Concatenate the datasets
                test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_i]) # Concatenate the datasets

        concat_datasets.append([train_dataset, val_dataset, test_dataset]) # Append the datasets to the joint dataset
        datasets = concat_datasets # Set the datasets to the joint dataset
    else:
        path_file = f"./results/{args.exp_name}/naive_fine-tuning_{args.dataset}.xlsx"
    
    workbook = xlsxwriter.Workbook(path_file) # Create the excel file
    test_acc_final = [] # List to save the test accuracy of each task and the test average accuracy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model
    if args.dataset == "mnist":
        model = Net_mnist().to(device) 
    elif args.dataset == "cifar10":
        model = Net_cifar10().to(device) 
    elif args.dataset == "cifar100":
        model = Net_cifar100().to(device)

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
        
        for epoch in range(args.epochs):
            print("="*100)
            if joint_datasets:
                print(f"METHOD: Joint-training -> Train on task: {id_task+1}, Epoch: {epoch}")
            else:
                print(f"METHOD: Fine-tuning -> Train on task: {id_task+1}, Epoch: {epoch}")
            
            # Training
            train_loss_epoch = train_epoch(model, device, train_loader, optimizer, id_task+1)

            # Validation 
            val_loss_epoch = val_epoch(model, device, val_loader, id_task+1)

            # Test
            test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy = test_epoch(model, 
                                                                                           device, 
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
                    model.load_state_dict(model_best.state_dict())
            
            # Save the results of the epoch if it is the last epoch
            if epoch == args.epochs-1:
                # Append the test accuracy of each task and the test average accuracy
                test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 

            print(f"Learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience}")

        if not joint_datasets:
            # Save the model
            save_model(model_best, args, id_task+1, method="fine-tuning", joint_datasets=False)

            # Save the results of the task
            save_training_results(dicc_results, workbook, id_task+1, training_name="fine-tuning")
        else:
            # Save the model
            save_model(model_best, args, id_task+1, method="joint-datasets", joint_datasets=True)

            # Save the results of the task
            save_training_results(dicc_results, workbook, id_task+1, training_name="joint-datasets")

    # Close the excel file
    workbook.close()

    return test_acc_final


def train_epoch(model, device, train_loader, optimizer, id_task):

    # Training
    model.train() # Set the model to training mode

    train_loss_accum = 0 # Training loss

    for images, targets in train_loader:
        # Move tensors to the configured device
        images = images.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad() 

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        train_loss = F.cross_entropy(outputs, targets)
        train_loss_accum += train_loss.item()

        # Backward pass
        train_loss.backward()

        # Optimize
        optimizer.step()
   
    train_loss_epoch = train_loss_accum/len(train_loader) # Training loss

    # Print the metrics
    print(f"Train on task {id_task} -> Loss: {train_loss_epoch}")

    return train_loss_epoch




def val_epoch(model, device, val_loader, id_task):

    # Validation
    model.eval() # Set the model to evaluation mode

    val_loss_epoch = 0
            
    with torch.no_grad():
        for images, targets in val_loader:
            # Move tensors to the configured device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            val_loss_epoch += F.cross_entropy(outputs, targets).item()

    val_loss_epoch /= len(val_loader) # Validation loss

    # Print the metrics
    print(f"Validation on task {id_task} -> Loss: {val_loss_epoch}")

    return val_loss_epoch



def test_epoch(model, device, datasets, args):

    # Test
    avg_accuracy = 0 # Average accuracy

    test_tasks_id = [] # List to save the results of the task
    test_tasks_loss = [] # List to save the test loss
    test_tasks_accuracy = [] # List to save the test accuracy

    model.eval() # Set the model to evaluation mode

    for id_task, task in enumerate(datasets):

        # Metrics for the test task 
        test_loss, correct_pred, accuracy = 0, 0, 0

        _, _, test_dataset = task # Get the images and labels from the task

        # Make the dataloader
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False)
        # Disable gradient calculation
        with torch.no_grad():
            for images, targets in test_loader:
                # Move tensors to the configured device
                images = images.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate the loss
                test_loss += F.cross_entropy(outputs, targets).item()

                # Get the index of the max log-probability
                pred = torch.argmax(outputs, dim=1)
                
                # Update the number of correct predictions
                correct_pred += torch.sum(pred == targets).item()

            # Calculate the average loss
            test_loss /= len(test_loader.dataset)

            # Calculate the average accuracy
            accuracy = 100. * correct_pred / len(test_loader.dataset)
            avg_accuracy += accuracy

            # Append the results to the lists
            test_tasks_id.append(id_task+1)
            test_tasks_loss.append(test_loss)
            test_tasks_accuracy.append(accuracy)

            # Print the metrics
            print(f"Test on task {id_task+1}: Average loss: {test_loss:.6f}, " 
                  f"Accuracy: {accuracy:.2f}%")
    
    # Calculate the average accuracy
    avg_accuracy /= len(datasets)
    print(f"Average accuracy: {avg_accuracy:.2f}%")

    return test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy

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