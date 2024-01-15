import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import xlsxwriter
import os
import sys
import copy
import numpy as np

sys.path.append('../')
from utils.save_training_results import save_training_results
from utils.utils import save_model
from models.architectures.net_mnist import Net_mnist
from models.architectures.net_cifar10 import Net_cifar10
from models.architectures.net_cifar100 import Net_cifar100

import wandb

def bimeco_training(datasets, args, config):

    print("\n")
    print("="*100)
    print("Training on BiMeCo (Bilateral Memory Consolidation) approach...")
    print("="*100)

    path_file = f'./results/{args.exp_name}/BiMeCo_{args.dataset}.xlsx'
    workbook = xlsxwriter.Workbook(path_file)  # Create the excel file
    test_acc_final = []  # List to save the average accuracy of each task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exemplar_set_img = []  # List to save the exemplar set
    exemplar_set_label = []  # List to save the exemplar set labels
    
    # Create the excel file
    if args.dataset == "mnist":
        model = Net_mnist().to(device)  # Instantiate the model
        num_classes = 10
        img_size = 28
        img_channels = 1
        feature_dim = 320
    elif args.dataset == "cifar10":
        model = Net_cifar10().to(device)  # Instantiate the model
        num_classes = 10
        img_size = 32
        img_channels = 3
        feature_dim = 512
    elif args.dataset == "cifar100" or args.dataset == "cifar100_alternative_dist":
        model = Net_cifar100().to(device)  # Instantiate the model
        num_classes = 100
        img_size = 32
        img_channels = 3
        feature_dim = 64

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
                print(f"METHOD: BiMeCo -> Train on task {id_task+1}, Epoch: {epoch+1}")

                # Training
                train_loss_epoch = normal_train(model, optimizer, train_loader, device)

                # Validation
                val_loss_epoch = normal_val(model, val_loader, device)

                # Test
                test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy = test(model, datasets, device, args)

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
            path_model = (f"./models/models_saved/{args.exp_name}/BiMeCo_{args.dataset}/"
                          f"BiMeCo_aftertask_{id_task}_{args.dataset}.pt")
            
            # Load model for short term memory
            model_short = copy.deepcopy(model) 
            model_short.load_state_dict(torch.load(path_model))

            # Load model for long term memory
            model_long = copy.deepcopy(model)
            model_long.load_state_dict(torch.load(path_model))

            # Create an optimizer for the short term memory model
            optimizer_short = optim.Adam(model_short.parameters(), lr=args.lr)  # Instantiate the optimizer
            optimizer_long = optim.Adam(model_long.parameters(), lr=args.lr)  # Instantiate the optimizer

            # Compute the ratio of current task samples to previous task samples
            ratio = len(tasks_dict[id_task]) / (len(tasks_dict[id_task]) + sum([len(tasks_dict[i]) for i in range(id_task)])) 
            
            train_dataloader_s = train_loader
            train_dataloader_l = torch.utils.data.DataLoader(dataset=train_dataset,
                                                                # batch_size=args.batch_size,
                                                                batch_size=int(ratio*args.batch_size), # Same ratio as the paper
                                                                shuffle=True)
            
            train_loss_epoch = 0
            total_epoch_loss_short, total_epoch_loss_long = 0, 0
            total_output_short, total_output_long = 0, 0
            total_diff_images_l, total_diff_images_s = 0, 0

            data_loader_exem_iter = iter(data_loader_exem)
            train_dataloader_l_iter = iter(train_dataloader_l)

            for epoch in range(args.epochs):
                print("="*100)
                print(f"METHOD: BiMeCo -> Train on task {id_task+1}, Epoch: {epoch+1}")
                
                # Sample a batch of data from train_dataloader_s
                for images_s, labels_s in train_dataloader_s:

                    # Sample two different batches from data_loader_exem
                    try:
                        images_exem_1, labels_exem_1 = next(data_loader_exem_iter)
                        images_exem_2, labels_exem_2 = next(data_loader_exem_iter)
                    except StopIteration:
                        data_loader_exem = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(tensor_exem_img, tensor_exem_label),
                                                                    batch_size=args.batch_size,
                                                                    shuffle=True)
                        data_loader_exem_iter = iter(data_loader_exem)
                        images_exem_1, labels_exem_1 = next(data_loader_exem_iter)
                        images_exem_2, labels_exem_2 = next(data_loader_exem_iter)

                    # Concatenate the images and labels of the first batch from data_loader_exem
                    images_s = torch.cat((images_s, images_exem_1), dim=0).to(device)
                    labels_s = torch.cat((labels_s, labels_exem_1), dim=0).to(device)

                    # Randomly sample ratio * batch_size samples from train_dataloader_l
                    try:
                        images_l, labels_l = next(train_dataloader_l_iter)
                    except StopIteration:
                        train_dataloader_l = torch.utils.data.DataLoader(dataset=train_dataset,
                                                                        #  batch_size=args.batch_size,
                                                                        batch_size=int(ratio*args.batch_size), # Same ratio as the paper
                                                                        shuffle=True)
                        train_dataloader_l_iter = iter(train_dataloader_l)
                        images_l, labels_l = next(train_dataloader_l_iter)

                    # Concatenate the images and labels of the second batch from data_loader_exem
                    images_l = torch.cat((images_l, images_exem_2), dim=0).to(device)
                    labels_l = torch.cat((labels_l, labels_exem_2), dim=0).to(device)

                    # Forward pass
                    epoch_loss_short, epoch_loss_long, output_short, output_long, diff_images_l, diff_images_s = (
                                                                    bimeco_train(model_short, model_long, optimizer_short, optimizer_long, 
                                                                     images_s, labels_s, images_l, labels_l, config)
                                                                     )
                    total_epoch_loss_short += epoch_loss_short
                    total_epoch_loss_long += epoch_loss_long
                    total_output_short += output_short
                    total_output_long += output_long
                    total_diff_images_l += diff_images_l
                    total_diff_images_s += diff_images_s

                # train_loss_epoch = epoch_loss_short + epoch_loss_long
                train_loss_epoch = total_epoch_loss_long
                print(f"Train loss: {total_epoch_loss_long}")
                print(f"Train loss output short: {total_output_short * config['bimeco_lambda_short']}")
                print(f"Train loss output long: {total_output_long * config['bimeco_lambda_long']}")
                print(f"Train loss diff images s: {total_diff_images_s}")
                print(f"Train loss diff images l: {total_diff_images_l}")
                print(f"Sum diff images: {(total_diff_images_l + total_diff_images_s)*config['bimeco_lambda_diff']}")

                # Update the parameters of the long term memory model
                for param_l, param_s in zip(model_long.parameters() ,  model_short.parameters()):
                    param_l.data = args.m * param_l.data + (1 - args.m) * param_s.data

                # Validation
                # val_loss_epoch = normal_val(model_long, val_loader, device)
                val_loss_epoch = bimeco_val(model_short, model_long, val_loader, device)

                # Test
                test_tasks_id, test_tasks_loss, test_tasks_accuracy, avg_accuracy = test(model_long, datasets, device, args)

                # Append the results to dicc_results
                wandb.log({"Task 2/Epoch": epoch+1, 
                           "Task 2/Train loss": train_loss_epoch,
                           "Task 2/Train loss output short": total_output_short * config['bimeco_lambda_short'],
                            "Task 2/Train loss output long": total_output_long * config['bimeco_lambda_long'],
                            "Task 2/Sum diff images": (total_diff_images_l + total_diff_images_s)*config['bimeco_lambda_diff'],
                            "Task 2/Validation loss": val_loss_epoch, 
                            "Task 2/Test loss task 1": test_tasks_loss[0],
                            "Task 2/Test loss task 2": test_tasks_loss[1], 
                            "Task 2/Test accuracy task 1": test_tasks_accuracy[0],
                            "Task 2/Test accuracy task 2": test_tasks_accuracy[1],
                            "Task 2/Test average accuracy": avg_accuracy})
                
                # Early stopping
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    patience = args.lr_patience
                    model_best = copy.deepcopy(model_long)

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
                        model_long.load_state_dict(model_best.state_dict())

                print(f"Current learning rate: {optimizer.param_groups[0]['lr']}, Patience: {patience}")

                # Save the results of the epoch if it is the last epoch
                if epoch == args.epochs-1:
                    test_acc_final.append([test_tasks_accuracy, avg_accuracy])

        # Save the results of the training
        save_training_results(dicc_results, workbook, id_task+1, training_name="BiMeCo")

        # Save the model
        save_model(model_best, args, id_task+1, method="BiMeCo")

        # Update memory buffer
        if id_task != args.num_tasks-1:
            exemplar_set_img, exemplar_set_label, tasks_dict  = after_train(model, exemplar_set_img, exemplar_set_label, train_dataset, 
                                                            device, id_task, args, img_channels, img_size, feature_dim, num_classes)

            tensor_exem_img = torch.empty((0, img_channels, img_size, img_size)) # Tensor to save the exemplar set
            tensor_exem_label = torch.empty((0), dtype=torch.long) # Tensor to save the exemplar set labels

            for index in range(len(exemplar_set_img)):
                tensor_exem_img = torch.cat((tensor_exem_img, torch.stack(exemplar_set_img[index])), dim=0) # Add the exemplar set to the tensor
                tensor_exem_label = torch.cat((tensor_exem_label, torch.stack(exemplar_set_label[index])), dim=0) # Add the exemplar set labels to the tensor

            # Make the dataloader
            data_loader_exem = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(tensor_exem_img, tensor_exem_label),
                                                           batch_size=args.batch_size,
                                                           shuffle=True)
        

    workbook.close()  # Close the excel file

    return test_acc_final


def normal_train(model, optimizer, data_loader, device):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)

def bimeco_train(model_short, model_long, optimizer_short, optimizer_long, images_s, labels_s, images_l, labels_l, config):

    model_short.train()
    model_long.train()

    # Training the short and long term memory models jointly
    optimizer_short.zero_grad()
    optimizer_long.zero_grad()

    # Get the outputs of the models
    output_short = model_short(images_s) 
    output_long = model_long(images_l)

    # Compute the difference between the feature extractor outputs
    feat_ext_short_model_images_s = F.normalize(model_short.feature_extractor(images_s))
    feat_ext_long_model_images_s = F.normalize(model_long.feature_extractor(images_s))
    diff_images_s = (feat_ext_short_model_images_s - feat_ext_long_model_images_s) ** 2 

    feat_ext_short_model_images_l = F.normalize(model_short.feature_extractor(images_l))
    feat_ext_long_model_images_l = F.normalize(model_long.feature_extractor(images_l))
    diff_images_l = (feat_ext_short_model_images_l - feat_ext_long_model_images_l) ** 2 

    diff = torch.cat((diff_images_s, diff_images_l), dim=0) # Concatenate the differences

    # Compute the loss
    loss = (config["bimeco_lambda_short"] * F.cross_entropy(output_short, labels_s) + 
            config["bimeco_lambda_long"] * F.cross_entropy(output_long, labels_l) + 
            config["bimeco_lambda_diff"] * diff.sum()) 
    
    epoch_loss_short = loss.item()
    epoch_loss_long = loss.item()
    output_short = F.cross_entropy(output_short, labels_s).item()
    output_long = F.cross_entropy(output_long, labels_l).item()
    diff_images_s = diff_images_s.sum().item() 
    diff_images_l = diff_images_l.sum().item()
    
    loss.backward() # Backward pass

    optimizer_short.step() # Update the parameters of the short term memory model
    optimizer_long.step() # Update the parameters of the long term memory model

    return epoch_loss_short, epoch_loss_long, output_short, output_long, diff_images_l, diff_images_s

def bimeco_val(model_short, model_long, data_loader, device):
    model_long.eval()
    loss = 0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            output_long = model_long(input)

            loss += F.cross_entropy(output_long, target)

    print(f"Val loss: {loss / len(data_loader)}")
    return loss / len(data_loader)

def normal_val(model, data_loader, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss += F.cross_entropy(output, target)


    print(f"Val loss: {loss / len(data_loader)}")
    return loss / len(data_loader)

def test(model, datasets, device, args):
    # Test
    model.eval()

    avg_acc = 0  # Average accuracy

    test_task_list = []  # List to save the results of the task
    test_loss_list = []  # List to save the test loss
    test_acc_list = []  # List to save the test accuracy

    for id_task_test, task in enumerate(datasets):
        test_loss, correct, accuracy = 0, 0, 0

        _, _, test_dataset = task  # Get the images and labels from the task

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False)
        with torch.no_grad():
            for input, target in test_loader:
                input, target = input.to(device), target.to(device)
                output = model(input)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).sum().item()

        test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct / len(test_loader.dataset)
        avg_acc += accuracy

        test_task_list.append(id_task_test+1)
        test_loss_list.append(test_loss)
        test_acc_list.append(accuracy)

        print(f"Test on task {id_task_test+1}: Average loss: {test_loss:.6f}, "
              f"Accuracy: {accuracy:.2f}%")

    avg_acc /= len(datasets)
    print(f"Average accuracy: {avg_acc:.2f}%")

    return test_task_list, test_loss_list, test_acc_list, avg_acc


def after_train(model, exemplar_set_img, exemplar_set_label, train_dataset, device, id_task, args,
                img_channels, img_size, feature_dim, num_classes):
    """
    Construct exemplar sets for each task using the iCaRL strategy.
    """
    print("="*100)
    print(f"METHOD: BiMeCo -> Update the exemplar set of task {id_task+1}")
    print("="*100)

    model.eval() # Set the model to evaluation mode

    m = int(args.memory_size / num_classes)  # Number of exemplars per class

    # Reduce exemplar set to the maximum size
    exemplar_set_img = [cls[:m] for cls in exemplar_set_img]
    exemplar_set_label = [cls[:m] for cls in exemplar_set_label]
    print(f"Size of class {index} exemplar: {len(exemplar_set_img[index])}" for index in range(len(exemplar_set_img)))

    if args.dataset == "cifar100_alternative_dist":
        # Create the tasks dictionary to know the classes of each task
        list_tasks = [80,100]
    else:
        # Create the tasks dictionary to know the classes of each task
        list_tasks = [num_classes // args.num_tasks * i for i in range(1, args.num_tasks + 1)]
        list_tasks[-1] = min(num_classes, list_tasks[-1])
    tasks_dict = {i: list(range(list_tasks[i-1] if i > 0 else 0, list_tasks[i])) for i in range(args.num_tasks)}

    # Take the classes of the current task 
    classes_task = [cls for cls in tasks_dict[id_task]]
    print(f"Creating the exemplar set for classes: {classes_task}")

    # Create the exemplar set
    images_ex = torch.empty((0, img_channels, img_size, img_size))
    labels_ex = torch.empty((0), dtype=torch.long)

    # Assuming train_dataset is a list of tuples (image, label)
    train_images, train_labels = zip(*train_dataset)
    train_images = torch.stack(train_images)
    train_labels = torch.stack(train_labels)

    for class_index in classes_task:
        exit = False
        selected_indexes = torch.where(train_labels == class_index)[0]
        images_ex = torch.cat((images_ex, train_images[selected_indexes]), dim=0)
        labels_ex = torch.cat((labels_ex, train_labels[selected_indexes]), dim=0)

        # Construct the exemplar set
        feature_extractor_output = F.normalize(model.feature_extractor(images_ex.to(device))).cpu().detach().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)

        exemplar_img = [] # List to save the exemplar set
        exemplar_label = [] # List to save the exemplar set labels
        now_class_mean = np.zeros((1, feature_dim)) # Current class mean

        selected_indexes = set() # Set to save the selected indexes
        # Construct the exemplar set
        for k in range(m):
            x = class_mean - (feature_extractor_output + now_class_mean) / (k + 1) # Equation 4
            x = np.linalg.norm(x, axis=1)

            index = np.argmin(x) # Equation 5  

            if index in selected_indexes:
                while index in selected_indexes:
                    if len(x) == len(selected_indexes):
                        print(f"All the images have been selected for class {class_index}")
                        exit = True
                        break
                    x[index] = np.inf
                    index = np.argmin(x)
                selected_indexes.add(index)
            
            if exit:
                break

            now_class_mean += feature_extractor_output[index] # Update the current class mean
            exemplar_img.append(images_ex[index]) # Add the exemplar to the exemplar set
            exemplar_label.append(labels_ex[index]) # Add the exemplar label to the exemplar set labels
            selected_indexes.add(index) # Add the index to the set

        exemplar_set_img.append(exemplar_img) # Add the exemplar set to the exemplar set list
        exemplar_set_label.append(exemplar_label) # Add the exemplar set labels to the exemplar set labels list
        images_ex = torch.empty((0, img_channels, img_size, img_size))
        labels_ex = torch.empty((0), dtype=torch.long) # Reset the labels variable

    print(f"Number of exemplars per class: {m}")
    print(f"Number of classes in the current task: {len(classes_task)}")
    print(f"Total size of the exemplar set images: {sum([len(exemplar_set_img[i]) for i in range(len(exemplar_set_img))])}")

    return exemplar_set_img, exemplar_set_label, tasks_dict 
