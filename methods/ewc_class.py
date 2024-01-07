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
    def __init__(self, current_model: nn.Module, old_model: nn.Module,
                 dataset: list, args: argparse.Namespace):

        self.current_model = current_model
        self.old_model = old_model
        self.dataset = dataset
        self.args = args

        self.params = {n: p for n, p in self.old_model.named_parameters() if p.requires_grad}
        self._precision_matrices = self._diag_fisher()
        self._means = {}

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}

        for n, p in deepcopy(self.params).items():
            p.data.zero_() # Make sure the precision matrice is empty
            precision_matrices[n] = variable(p.data)

        self.current_model.eval()
        for input, label in self.dataset:
            self.current_model.zero_grad()
            input = variable(input)
            label = variable(label)

            output = self.current_model(input)
            loss = F.cross_entropy(output, label)
            loss.backward()

            for n, p in self.current_model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / self.args.batch_size           

        # for input, _ in self.dataset:
        #     self.model.zero_grad()
        #     input = variable(input)
        #     output = self.model(input).view(1, -1)
        #     label = output.max(1)[1].view(-1)
        #     loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        #     loss.backward()

        #     for n, p in self.model.named_parameters():
        #         precision_matrices[n].data += p.grad.data ** 2 / args.batch_size

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
    loss = 0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = variable(input), variable(target)
            output = model(input)
            loss += F.cross_entropy(output, target)

    print(f"Val loss: {loss / len(data_loader)}")    
    return loss / len(data_loader)


def ewc_train(current_model: nn.Module, optimizer: torch.optim, 
              data_loader: torch.utils.data.DataLoader, ewc: EWC, importance: float):
    current_model.train()
    epoch_loss = 0
    ce_loss = 0
    ewc_loss = 0

    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = current_model(input)

        ce_loss += F.cross_entropy(output, target)
        ewc_loss += importance * ewc.penalty(current_model)

        loss = F.cross_entropy(output, target) + importance * ewc.penalty(current_model)
        
        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    print(f"CE loss: {ce_loss / len(data_loader)}")
    print(f"EWC loss: {ewc_loss / len(data_loader)}")

    return epoch_loss / len(data_loader)

def ewc_validate(current_model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                 ewc: EWC, importance: float):
    current_model.eval()
    loss = 0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = variable(input), variable(target)
            output = current_model(input)
            loss += F.cross_entropy(output, target) + importance * ewc.penalty(current_model)

    print(f"Val loss: {loss / len(data_loader)}")
    return loss / len(data_loader)


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
        with torch.no_grad():
            for input, target in test_loader:
                input, target = variable(input), variable(target)
                output = model(input)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).sum().item()

        test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct / len(test_loader.dataset)
        avg_acc += accuracy

        test_task_list.append(id_task_test)
        test_loss_list.append(test_loss)
        test_acc_list.append(accuracy)

        print(f"Test on task {id_task_test+1}: Average loss: {test_loss:.6f}, "
              f"Accuracy: {accuracy:.2f}%")

    avg_acc /= len(datasets)
    print(f"Average accuracy: {avg_acc:.2f}%")


    return test_task_list, test_loss_list, test_acc_list, avg_acc