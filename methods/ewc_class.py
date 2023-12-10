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


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input, _ in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


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


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)

def ewc_validate(model: nn.Module, data_loader: torch.utils.data.DataLoader, ewc: EWC, importance: float):
    model.eval()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.data.item()

    print(f"Val loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)


def test(model: nn.Module, datasets: list, args: argparse.Namespace):
    # Test
    model.eval()

    avg_acc = 0 # Average accuracy

    test_task_list = [] # List to save the results of the task
    test_loss_list = [] # List to save the test loss
    test_acc_list = [] # List to save the test accuracy


    for id_task_test, task in enumerate(datasets):
        test_loss, correct, accuracy = 0, 0, 0

        _, _, test_dataset = task # Get the images and labels from the task

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

        print(f"Test on task {id_task_test+1}: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%")

    avg_acc /= len(datasets)
    print(f"Average accuracy: {avg_acc:.2f}%")


    return test_task_list, test_loss_list, test_acc_list, avg_acc