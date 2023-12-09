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

def lwf_training(datasets, args):
    
    print("------------------------------------------")
    print("Training on Learning without Forgetting (LwF) approach...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the excel file
    if args.dataset == "mnist":
        path_file = "./results/mnist/results_mnist_lwf.xlsx"
        model = Net_mnist().to(device) # Instantiate the model

    elif args.dataset == "cifar10":
        path_file = "./results/cifar10/results_cifar10_lwf.xlsx"
        model = Net_cifar10().to(device) # Instantiate the model
        
    elif args.dataset == "cifar100":
        path_file = "./results/cifar100/results_cifar100_lwf.xlsx"
        model = Net_cifar100().to(device) # Instantiate the model

    # Remove previous models saved if they exist
    for i in range(len(datasets)):
        if os.path.exists(f'./models/lwf_{args.dataset}_aftertask_{i+1}.pth'):
            os.remove(f'./models/lwf_{args.dataset}_aftertask_{i+1}.pth')
    
    if os.path.exists(path_file):
        os.remove(path_file)
    workbook = xlsxwriter.Workbook(path_file)

    avg_acc_list = []

    for id_task_dataset, task in enumerate(datasets):

        optimizer = optim.Adam(model.parameters(), lr=args.lr) # Instantiate the optimizer     

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, 
                                                gamma=args.scheduler_gamma)  

        dicc_results = {"Train task":[], "Train epoch": [], "Train loss":[], "Val loss":[],
                         "Test task":[], "Test loss":[], "Test accuracy":[], "Test average accuracy": []}
        print("------------------------------------------")

        train_dataset, val_dataset, _ = task
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
        
        for epoch in range(args.epochs):
            print("------------------------------------------")

            train_loss_epoch = train_epoch(model, device, train_loader, optimizer, id_task_dataset, epoch, args, scheduler)

            val_loss_epoch = val_epoch(model, device, val_loader, id_task_dataset, epoch)

            test_task_list, test_loss_list, test_acc_list, avg_acc = test_epoch(model, device, datasets, args)

            dicc_results["Train task"].append(id_task_dataset+1)
            dicc_results["Train epoch"].append(epoch+1)
            dicc_results["Train loss"].append(train_loss_epoch)
            dicc_results["Val loss"].append(val_loss_epoch)
            dicc_results["Test task"].append(test_task_list)
            dicc_results["Test loss"].append(test_loss_list)
            dicc_results["Test accuracy"].append(test_acc_list)
            dicc_results["Test average accuracy"].append(avg_acc)

            if epoch == args.epochs-1:
                avg_acc_list.append(avg_acc)
            
        # Save the model
        if id_task_dataset+1 < len(datasets):
            torch.save(model.state_dict(), f'./models/lwf_{args.dataset}_aftertask_{id_task_dataset+1}.pth')

        save_training_results(dicc_results, workbook, task=id_task_dataset, training_name="LwF")
    
    workbook.close()

    return avg_acc_list


def train_epoch(model, device, train_loader, optimizer, id_task_dataset, epoch, args, scheduler):

    # Training
    model.train()

    train_loss_acc = 0 # Training loss
    distillation_loss_value = 0 # Distillation loss

    if id_task_dataset+1 > 1: # Apply LwF from the second task onward
        if args.dataset == "mnist":
            prev_model = Net_mnist().to(device)
        elif args.dataset == "cifar10":
            prev_model = Net_cifar10().to(device)
        elif args.dataset == "cifar100":
            prev_model = Net_cifar100().to(device)

        # Load the previous model
        prev_model.load_state_dict(torch.load(f'./models/lwf_{args.dataset}_aftertask_{id_task_dataset}.pth')) 
    
    print(f"Training on task {id_task_dataset+1}...")

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # Clear the gradients

        outputs = model(images) # Forward pass

        train_loss = F.cross_entropy(outputs, labels) # Calculate the loss
    
        # Distillation loss (LwF)
        if id_task_dataset+1 > 1:  # Apply LwF from the second task onward

            prev_model.eval() # Set the previous model to evaluation mode
            
            prev_outputs = prev_model(images) # Forward pass with the previous model

            distillation_loss = F.kl_div(F.log_softmax(outputs, dim=1), 
                                        F.softmax(prev_outputs, dim=1)) # Calculate the distillation loss
            distillation_loss_value += distillation_loss.item() # Accumulate the distillation loss

            train_loss += args.lwf_lambda * distillation_loss # Add the distillation loss to the training loss

        train_loss_acc += train_loss.item() # Accumulate the training loss
        train_loss.backward() # Backward pass
        optimizer.step() # Optimize

    scheduler.step() # Update the learning rate
    print(f"Epoch: {epoch+1}, Learning rate: {scheduler.get_last_lr()[0]}")


    train_loss_epoch = train_loss_acc/len(train_loader) # Training loss
    distillation_loss_value /= len(train_loader) # Distillation loss

    print(f"Trained on task {id_task_dataset+1} -> Epoch: {epoch+1}, Loss: {train_loss_epoch}")
    if id_task_dataset+1 > 1:
        print(f"Distillation loss: {distillation_loss_value} // Lambda: {args.lwf_lambda}")

    return train_loss_epoch



def val_epoch(model, device, val_loader, id_task_dataset, epoch):
     
    # Validation
    model.eval() # Set the model to evaluation mode

    val_loss_epoch = 0 # Validation loss

    with torch.no_grad():
        for images, labels in val_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            val_loss = F.cross_entropy(outputs, labels)
            val_loss_epoch += val_loss.item()
    
    val_loss_epoch /= len(val_loader) # Validation loss

    # Print the metrics
    print(f"Validated on task {id_task_dataset + 1} -> Epoch: {epoch+1}, Loss: {val_loss_epoch}")

    return val_loss_epoch



def test_epoch(model, device, datasets, args):
    """
    Test the model.
    """


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


