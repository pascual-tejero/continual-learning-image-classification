import torch
import torch.nn.functional as F
import torch.optim as optim

import xlsxwriter
import sys
import copy

sys.path.append('../')
from utils.save_training_results import save_training_results
from utils.utils import save_model

from models.architectures.net_mnist import Net_mnist
from models.architectures.net_cifar10 import Net_cifar10
from models.architectures.net_cifar100 import Net_cifar100

def rehearsal_training(datasets, args, rehearsal_prop, random_rehearsal=False):
    """
    In this function, we train the model using the rehearsal approach.

    :param datasets: list of datasets
    :param args: arguments from the command line
    :param rehearsal_prop: percentage of rehearsal data
    :param random_rehearsal: if True, rehearse randomly

    :return: test_acc_final: list with the test accuracy of each task and the test average accuracy

    """
    rehearsal_perc = int(rehearsal_prop*100) # Percentage of rehearsal data
    print("\n")
    print("="*100)
    print(f"Training: REHEARSAL approach, percentage: {rehearsal_perc}%...")
    print("="*100)

    path_file = f"./results/{args.exp_name}/rehearsal{rehearsal_perc}%_{args.dataset}.xlsx"
    workbook = xlsxwriter.Workbook(path_file)  # Create the excel file
    test_acc_final = [] # List to save the test accuracy of each task and the test average accuracy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed) # Set the seed

    # Create model
    if args.dataset == "mnist":
        model = Net_mnist().to(device) # Instantiate the mod
    elif args.dataset == "cifar10":
        model = Net_cifar10().to(device) # Instantiate the model
    elif args.dataset == "cifar100" or args.dataset == "cifar100-alternative-dist":
        model = Net_cifar100().to(device) # Instantiate the model
        
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

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

        # Implement rehearsal by combining previous task data with the current task data
        if id_task > 0:
            # Make the dataloader for the rehearsal data
            rehearsal_data_train, rehearsal_data_val = add_prev_tasks_to_current_task(datasets, 
                                                                                      id_task, 
                                                                                      rehearsal_prop, 
                                                                                      random_rehearsal)
        else:
            rehearsal_data_train, rehearsal_data_val, _ = task  # Get the images and labels from the task

        train_loader = torch.utils.data.DataLoader(dataset=rehearsal_data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
        
        val_loader = torch.utils.data.DataLoader(dataset=rehearsal_data_val,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)

        for epoch in range(args.epochs):
            print("="*100)
            print(f"METHOD: Rehearsal training {rehearsal_perc}% (Experiment: {args.exp_name}) "
                   f"-> Train on task {id_task+1} -> Epoch: {epoch+1}")
            
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

        # Save the results of the task
        save_training_results(dicc_results, workbook, id_task+1, 
                              training_name=f"rehearsal{rehearsal_perc}%")

        # Save the best model after each task
        save_model(model_best, args, id_task+1, method=f"rehearsal{rehearsal_perc}%")

    # Close the excel file
    workbook.close()

    return test_acc_final


def add_prev_tasks_to_current_task(datasets, id_task, rehearsal_prop, random_rehearsal=True):
    """
    Add the previous datasets (lower ids) to the current dataset (higher id) to perform rehearsal.
    """
    previous_datasets = datasets[:id_task] # Get the previous datasets

    current_dataset = datasets[id_task] # Get the current dataset
    rehearsal_data_train = current_dataset[0] # Get the training data from the current dataset
    rehearsal_data_val = current_dataset[1] # Get the validation data from the current dataset

    for (train_set, val_set, _) in previous_datasets: # Iterate over the previous datasets
        num_samples_train = int(len(train_set) * rehearsal_prop) # Get the number of samples to rehearse
        num_samples_val = int(len(val_set) * rehearsal_prop) # Get the number of samples to rehearse

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

    print("="*100)
    print(f"Training a new task with rehearsal, percentage ({int(rehearsal_prop*100)}%) of rehearsal data")
    print(f"Adding previous data -> Rehearsal data: {len(rehearsal_data_train)} training samples data and "
            f"{len(rehearsal_data_val)} validation samples")
    
    return rehearsal_data_train, rehearsal_data_val
    


def train_epoch(model, device, train_loader, optimizer, id_task):

    model.train()  # Set the model to training mode

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
            
    val_loss_epoch = 0 # Validation loss

    with torch.no_grad():
        for images, targets in val_loader:
            # Move tensors to the configured device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            val_loss_epoch += F.cross_entropy(outputs, targets)

    val_loss_epoch /= len(val_loader) # Validation loss

    # Print the metrics
    print(f"Validation on task {id_task} -> Loss: {val_loss_epoch}")

    return val_loss_epoch


def test_epoch(model, device, datasets, args):

    # Test
    avg_accurracy = 0  # Average accuracy

    test_tasks_id = [] # List to save the results of the task
    test_tasks_loss = [] # List to save the test loss
    test_tasks_accuracy = [] # List to save the test accuracy

    for id_task, task in enumerate(datasets):

        # Metrics for the test task
        test_loss, correct_pred, accuracy = 0, 0, 0

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
                correct_pred += torch.sum(pred == labels).item()

            # Calculate the average loss
            test_loss /= len(test_loader.dataset)

            # Calculate the average accuracy
            accuracy = 100. * correct_pred / len(test_loader.dataset)
            avg_accurracy += accuracy

            # Append the results to the lists
            test_tasks_id.append(id_task+1)
            test_tasks_loss.append(test_loss)
            test_tasks_accuracy.append(accuracy)

            # Print the metrics
            print(f"Test on task {id_task+1}: Average loss: {test_loss:.6f}, " 
                    f"Accuracy: {accuracy:.2f}%")
    
    # Calculate the average accuracy
    avg_accurracy /= len(datasets)
    print(f"Average accuracy: {avg_accurracy:.2f}%")

    return test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accurracy


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
