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

def rehearsal_training(datasets, args, rehearsal_percentage, random_rehearsal=False):
    """
    In this function, we train the model using the rehearsal approach, which includes rehearsal of previous tasks.
    """
    print("------------------------------------------")
    print(f"Training with rehearsal approach with rehearsal percentage: {rehearsal_percentage*100}%...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the excel file
    if args.dataset == "mnist":
        path_file = f"./results/mnist/results_mnist_rehearsal_{rehearsal_percentage}.xlsx"
        model = Net_mnist().to(device) # Instantiate the mod
    elif args.dataset == "cifar10":
        path_file = f"./results/cifar10/results_cifar10_rehearsal_{rehearsal_percentage}.xlsx"
        model = Net_cifar10().to(device) # Instantiate the model
        
    elif args.dataset == "cifar100":
        path_file = f"./results/cifar100/results_cifar100_rehearsal_{rehearsal_percentage}.xlsx"
        model = Net_cifar100().to(device) # Instantiate the model

    optimizer = optim.Adam(model.parameters(), lr=args.lr) # Instantiate the optimizer  

    if os.path.exists(path_file):  # If the file exists
        os.remove(path_file)  # Remove the file if it exists
    workbook = xlsxwriter.Workbook(path_file)  # Create the excel file

    avg_acc_list = []  # List to save the average accuracy of each task

    for id_task_dataset, task in enumerate(datasets):
        
        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, 
                                                gamma=args.scheduler_gamma)
        
        dicc_results = {"Train task":[], "Train epoch": [], "Train loss":[], "Val loss":[],
                         "Test task":[], "Test loss":[], "Test accuracy":[], "Test average accuracy": []}
        print("------------------------------------------")

        # Implement rehearsal by combining previous task data with the current task data
        if id_task_dataset > 0:
            # Make the dataloader for the rehearsal data
            rehearsal_data_train, rehearsal_data_val = add_previous_datasets_to_current_dataset(datasets, 
                                                                                                id_task_dataset, 
                                                                                                rehearsal_percentage, 
                                                                                                random_rehearsal=True)
        else:
            rehearsal_data_train, rehearsal_data_val, _ = task  # Get the images and labels from the task

        train_loader = torch.utils.data.DataLoader(dataset=rehearsal_data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
        
        val_loader = torch.utils.data.DataLoader(dataset=rehearsal_data_val,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)

        for epoch in range(args.epochs):
            print("------------------------------------------")

            # Training
            train_loss_epoch = train_epoch(model, device, train_loader, optimizer, id_task_dataset, epoch, scheduler)

            # Validation
            val_loss_epoch = val_epoch(model, device, val_loader, id_task_dataset, epoch)

            # Test
            test_task_list, test_loss_list, test_acc_list, avg_acc = test_epoch(model, device, datasets, args)

            # Append the results to dicc_results
            dicc_results["Train task"].append(id_task_dataset+1)
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

        # Save the results
        save_training_results(dicc_results, workbook, task=id_task_dataset, training_name="rehearsal")

    # Close the excel file
    workbook.close()

    return avg_acc_list


def add_previous_datasets_to_current_dataset(datasets, id_task_dataset, rehearsal_percentage, random_rehearsal=True):
    """
    Add the previous datasets (lower ids) to the current dataset (higher id) to perform rehearsal.
    """
    previous_datasets = datasets[:id_task_dataset] # Get the previous datasets

    current_dataset = datasets[id_task_dataset] # Get the current dataset
    rehearsal_data_train = current_dataset[0] # Get the training data from the current dataset
    rehearsal_data_val = current_dataset[1] # Get the validation data from the current dataset

    for (train_set, val_set, _) in previous_datasets: # Iterate over the previous datasets
        num_samples_train = int(len(train_set) * rehearsal_percentage) # Get the number of samples to rehearse
        num_samples_val = int(len(val_set) * rehearsal_percentage) # Get the number of samples to rehearse

        if random_rehearsal: # If we want to rehearse randomly

            # Get the subset of data to rehearse
            subset_data_train = torch.utils.data.Subset(train_set, 
                                                        torch.randperm(len(train_set))[:num_samples_train]) 
            
            # Get the subset of data to rehearse
            subset_data_val = torch.utils.data.Subset(val_set, 
                                                      torch.randperm(len(val_set))[:num_samples_val])

        else: # If we want to rehearse sequentially

            # Get the subset of data to rehearse
            subset_data_train = torch.utils.data.Subset(train_set, torch.arange(num_samples_train))
            # Get the subset of data to rehearse
            subset_data_val = torch.utils.data.Subset(val_set, torch.arange(num_samples_val)) 

        rehearsal_data_train = torch.utils.data.ConcatDataset([rehearsal_data_train, subset_data_train]) # Concatenate the data
        rehearsal_data_val = torch.utils.data.ConcatDataset([rehearsal_data_val, subset_data_val]) # Concatenate the data

    print(f"Rehearsal data: {len(rehearsal_data_train)} training samples data and "
            f"{len(rehearsal_data_val)} validation samples")
    
    return rehearsal_data_train, rehearsal_data_val
    


def train_epoch(model, device, train_loader, optimizer, id_task_dataset, epoch, scheduler):

    model.train()  # Set the model to training mode

    train_loss_acc = 0 # Training loss


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
        train_loss_acc += train_loss.item()

        # Backward pass
        train_loss.backward()

        # Optimize
        optimizer.step()
    
    scheduler.step() # Update the learning rate
    print(f"Epoch: {epoch+1}, Learning rate: {scheduler.get_last_lr()[0]}")

    train_loss_epoch = train_loss_acc/len(train_loader) # Training loss

    # Print the metrics
    print(f"Trained on task {id_task_dataset + 1} -> Epoch: {epoch+1}, Loss: {train_loss_epoch}")

    return train_loss_epoch


def val_epoch(model, device, val_loader, id_task_dataset, epoch):
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
    print(f"Validated on task {id_task_dataset + 1} -> Epoch: {epoch+1}, Loss: {val_loss_value}")

    return val_loss_epoch


def test_epoch(model, device, datasets, args):

    # Test
    avg_acc = 0  # Average accuracy

    test_task_list = []  # List to save the results of the task
    test_loss_list = []  # List to save the test loss
    test_acc_list = []  # List to save the test accuracy

    for id_task_test, task in enumerate(datasets):
        # Metrics
        test_loss, correct, accuracy = 0, 0, 0

        _, _, test_dataset = task  # Get the images and labels from the task

        # Make the dataloader
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)

        # Set the model to evaluation mode
        model.eval()

        # Disable gradient calculation
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

                # Update the number of correct predictions
                correct += torch.sum(pred == labels).item()

            # Calculate the average loss
            test_loss /= len(test_loader.dataset)

        # Calculate the average accuracy
        accuracy = 100. * correct / len(test_loader.dataset)
        avg_acc += accuracy

        # Append the results to the lists
        test_task_list.append(id_task_test + 1)
        test_loss_list.append(test_loss)
        test_acc_list.append(accuracy)

        # Print the metrics
        print(f"Test on task {id_task_test + 1}: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%")
    
    # Calculate the average accuracy
    avg_acc /= len(datasets)
    print(f"Average accuracy: {avg_acc:.0f}%")

    return test_task_list, test_loss_list, test_acc_list, avg_acc


            

