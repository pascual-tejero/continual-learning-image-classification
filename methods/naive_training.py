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

def naive_training(datasets, args, joint_training=False):
    
    """
    In this function, we train the model using the naive approach, which is training the model on the first dataset
    and then training the model on the second dataset.
    """

    print("------------------------------------------")
    print("Training on naive approach...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if joint_training: # If we want to train the model on the joint dataset
        joint_dataset = []

        train_dataset, val_dataset, test_dataset = datasets[0] # Get the images and labels from the task

        if len(datasets) > 1: # If there are more than one task
            for i in range(1,len(datasets)): # For each task
                train_dataset_i, val_dataset_i, test_dataset_i = datasets[i] # Get the images and labels from the task
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_i]) # Concatenate the datasets
                val_dataset = torch.utils.data.ConcatDataset([val_dataset, val_dataset_i]) # Concatenate the datasets
                test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_i]) # Concatenate the datasets

        joint_dataset.append([train_dataset, val_dataset, test_dataset]) # Append the datasets to the joint dataset
        datasets = joint_dataset # Set the datasets to the joint dataset
 
    # Create the excel file
    if args.dataset == "mnist":
        model = Net_mnist().to(device) # Instantiate the model

    elif args.dataset == "cifar10":
        model = Net_cifar10().to(device) # Instantiate the model
        
    elif args.dataset == "cifar100":
        model = Net_cifar100().to(device) # Instantiate the model

    if joint_training:
        path_file = f"./results/{args.exp_name}/results_naive_joint-training_{args.dataset}.xlsx"
    else:
        path_file = f"./results/{args.exp_name}/results_naive_fine-tuning_{args.dataset}.xlsx"

    workbook = xlsxwriter.Workbook(path_file) # Create the excel file

    avg_acc_list = [] # List to save the average accuracy of each task

    for id_task_dataset, task in enumerate(datasets):

        patience = args.lr_patience # Patience for early stopping
        lr = args.lr # Learning rate
        best_val_loss = 1e20 # Validation loss of the previous epoch
        model_best = copy.deepcopy(model) # Save the best model so far

        optimizer = optim.Adam(model.parameters(), lr=args.lr) # Instantiate the optimizer     

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
            train_loss_epoch = train_epoch(model, device, train_loader, optimizer, id_task_dataset, epoch)

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

        # Save the results of the task
        save_training_results(dicc_results, workbook, task=id_task_dataset, training_name="naive")

        if not joint_training:
            # Save the model
            save_model(model_best, args, id_task_dataset+1, task="fine-tuning", joint_training=False)
        else:
            # Save the model
            save_model(model_best, args, id_task_dataset+1, task="joint-datasets", joint_training=True)

    # Close the excel file
    workbook.close()



    return avg_acc_list


def train_epoch(model, device, train_loader, optimizer, id_task_dataset, epoch):

    # Training
    model.train() # Set the model to training mode

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


    print(f"Epoch: {epoch+1}, Learning rate: {optimizer.param_groups[0]['lr']}")

    train_loss_epoch = train_loss_acc/len(train_loader) # Training loss

    # Print the metrics
    print(f"Trained on task {id_task_dataset+1} -> Epoch: {epoch+1}, Loss: {train_loss_epoch}")

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
    print(f"Validated on task {id_task_dataset+1} -> Epoch: {epoch+1}, Loss: {val_loss_epoch}")

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

