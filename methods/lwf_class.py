from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import argparse


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


  


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)


def normal_val(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.item()

    print(f"Val loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)


def lwf_train(model: nn.Module, old_model:nn.Module, optimizer: torch.optim, 
              data_loader: torch.utils.data.DataLoader, alpha: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)

        # Get the predictions of the current model
        current_predictions = F.log_softmax(model(input), dim=1)
        old_predictions = F.softmax(old_model(input), dim=1)
        
        # Calculate the KL divergence between the current and old predictions
        penalty = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

        loss = F.cross_entropy(output, target) + alpha * penalty 

        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    print(f"Penalty: {penalty.data.item()}")
    return epoch_loss / len(data_loader)


def lwf_validate(model: nn.Module, old_model:nn.Module, 
                 data_loader: torch.utils.data.DataLoader, alpha: float):
    model.eval()
    old_model.eval()
    epoch_loss = 0

    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        
        # Get the predictions of the current model
        current_predictions = F.log_softmax(model(input), dim=1)
        old_predictions = F.softmax(old_model(input), dim=1)
        
        # Calculate the KL divergence between the current and old predictions
        penalty = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

        loss = F.cross_entropy(output, target) + alpha * penalty  
        
        epoch_loss += loss.data.item()

    print(f"Val loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)


def test(model: nn.Module, datasets: list, args: argparse.Namespace):
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

        for input, target in test_loader:
            input, target = variable(input), variable(target)
            output = model(input)
            test_loss += F.cross_entropy(output, target, reduction="sum").data.item()
            correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()

        test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct / len(test_loader.dataset)
        avg_acc += accuracy

        test_task_list.append(id_task_test)
        test_loss_list.append(test_loss)
        test_acc_list.append(accuracy)

        print(f"Test on task {id_task_test + 1}: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%")

    avg_acc /= len(datasets)
    print(f"Average accuracy: {avg_acc:.2f}%")

    return test_task_list, test_loss_list, test_acc_list, avg_acc
