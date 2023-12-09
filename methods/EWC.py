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

    optimizer = optim.Adam(model.parameters(), lr=args.lr) # Instantiate the optimizer     
    
    if os.path.exists(path_file): # If the file exists
        os.remove(path_file) # Remove the file if it exists
    workbook = xlsxwriter.Workbook(path_file) # Create the excel file

    avg_acc_list = [] # List to save the average accuracy of each task

    fisher_dict = {}
    optpar_dict = {}

    for id_task, task in enumerate(datasets):
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
        
        for epoch in range(args.epochs):
            print("------------------------------------------")

            # Training
            train_loss_epoch = train_epoch(model, device, train_loader, optimizer, id_task, 
                                           fisher_dict, optpar_dict, epoch, args)
            

            # Validation 
            val_loss_epoch = val_epoch(model, device, val_loader, id_task, epoch)

            # Test
            test_task_list, test_loss_list, test_acc_list, avg_acc = test_epoch(model, device, datasets, args)


            # Append the results to dicc_results
            dicc_results["Train task"].append(id_task+1)
            dicc_results["Train epoch"].append(epoch+1)
            dicc_results["Train loss"].append(train_loss_epoch)
            dicc_results["Val loss"].append(val_loss_epoch)
            dicc_results["Test task"].append(test_task_list)
            dicc_results["Test loss"].append(test_loss_list)
            dicc_results["Test accuracy"].append(test_acc_list)
            dicc_results["Test average accuracy"].append(avg_acc)
            # print(dicc_results)

            if epoch == args.epochs-1:
                avg_acc_list.append(avg_acc)
        
        # Update the model with task-specific information
        on_task_update(id_task, model, fisher_dict, optpar_dict)  

        # Save the results
        save_training_results(dicc_results, workbook, task=id_task, training_name="EWC")
    
    # Close the excel file
    workbook.close()

    return avg_acc_list


def train_epoch(model, device, train_loader, optimizer, id_task, fisher_dict, optpar_dict, epoch, args):

    # Training
    model.train() # Set the model to training mode

    train_loss_acc = 0 # Training loss
    ce_train_loss = 0 # Cross-entropy training loss
    ewc_train_loss = 0 # EWC training loss

    for images, labels in train_loader:
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad() 

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        train_loss = F.cross_entropy(outputs, labels)
        ce_train_loss += train_loss.item()

        for task in range(id_task):

            for name, param in model.named_parameters():
                # Fisher information for the parameter
                fisher = fisher_dict[task][name]

                # Previous parameter value
                optpar = optpar_dict[task][name]

                # EWC loss is added to the original loss to prevent catastrophic forgetting.
                # It penalizes changes in model parameters based on the Fisher information
                # and previous parameter values
                train_loss += (fisher * (optpar - param).pow(2)).sum() * args.ewc_lambda
                ewc_train_loss += train_loss.item()

        train_loss_acc += train_loss.item() # Accumulate the training loss

        # Backward pass
        train_loss.backward()

        # Optimize
        optimizer.step()

    train_loss_epoch = train_loss_acc/len(train_loader) # Training loss

    # Print the metrics
    print(f"Trained on task {id_task+1} -> Epoch: {epoch+1}, Loss: {train_loss_epoch}")
    print(f"Cross-entropy loss: {ce_train_loss/len(train_loader)}")
    print(f"EWC loss: {ewc_train_loss/len(train_loader)}")

    return train_loss_epoch

def on_task_update(id_task, model, fisher_dict, optpar_dict):
    # Initialize dictionaries for Fisher information and model parameters
    fisher_dict[id_task] = {}
    optpar_dict[id_task] = {}

    # Calculate Fisher information for model parameters
    for name, param in model.named_parameters():

        # Store a clone of the model's parameter data
        optpar_dict[id_task][name] = param.data.clone()

        # Calculate Fisher information (squared gradients) for each parameter
        fisher_dict[id_task][name] = param.grad.data.clone().pow(2)


def val_epoch(model, device, val_loader, id_task, epoch):

    # Validation
    model.eval() # Set the model to evaluation mode
            
    val_loss_value = 0 # Validation loss

    with torch.no_grad():
        for images, labels in val_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            val_loss = F.cross_entropy(outputs, labels)
            val_loss_value += val_loss.item()

    val_loss_epoch = val_loss_value/len(val_loader) # Validation loss

    # Print the metrics
    print(f"Validated on task {id_task+1} -> Epoch: {epoch+1}, Loss: {val_loss_epoch}")

    return val_loss_epoch



def test_epoch(model, device, datasets, args):

    # Test
    avg_acc = 0 # Average accuracy

    test_task_list = [] # List to save the results of the task
    test_loss_list = [] # List to save the test loss
    test_acc_list = [] # List to save the test accuracy

    for id_task_test, task in enumerate(datasets):

        # Metrics
        test_loss, correct, accuracy = 0, 0, 0

        _, _, test_dataset = task # Get the images and labels from the task

        # Make the dataloader
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False)

        # Disable gradient calculation
        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            for images, labels in test_loader:
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate the loss
                test_loss += F.cross_entropy(outputs, labels).item()

                # Get the index of the max log-probability
                pred = torch.argmax(outputs, dim=1)
                # labels = torch.argmax(labels, dim=1)
                
                # Update the number of correct predictions
                correct += torch.sum(pred == labels).item()

            # Calculate the average loss
            test_loss /= len(test_loader.dataset)

        # Calculate the average accuracy
        accuracy = 100. * correct / len(test_loader.dataset)
        avg_acc += accuracy

        # Append the results to the lists
        test_task_list.append(id_task_test+1)
        test_loss_list.append(test_loss)
        test_acc_list.append(accuracy)

        # Print the metrics
        print(f"Test on task {id_task_test+1}: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%")
    
    # Calculate the average accuracy
    avg_acc /= len(datasets)
    print(f"Average accuracy: {avg_acc:.0f}%")

    return test_task_list, test_loss_list, test_acc_list, avg_acc