import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

import argparse
import numpy as np
import copy


class LWF(object):
    def __init__(self, model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad,
                 momentum, fix_bn, weight_decay, lamb, lambda_aux):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.fix_bn = fix_bn
        self.weight_decay = weight_decay
        self.lamb = lamb
        self.lambda_aux = lambda_aux
        self.optimizer = None
        self.model_old = None
        self.model_aux = None

    def _getoptimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def train_loop(self, model, task_id, train_loader, val_loader, test_loader,
                   old_model=None, aux_model=None):
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = copy.deepcopy(model) ###
        
        optimizer_main = self._getoptimizer() # Optimizer for the main network


        if task_id > 0:
            if aux_model is not None:
                best_model_aux = copy.deepcopy(best_model) ### 

                optimizer_aux = self._getoptimizer() # Optimizer for the auxiliar network

                for epoch in range(self.nepochs): 
                    train_loss = self.train_epoch(aux_model, task_id, train_loader, optimizer_aux)

                    val_loss = self.validate_epoch(aux_model, task_id, val_loader)

                    test_loss, test_acc = self.test(task_id, test_loader)
            
                    aux_model, best_val_loss, patience, exit_training = self.adjust_lr_patience(aux_model, 
                                                                            optimizer_aux, best_val_loss, 
                                                                            val_loss, patience)

                    if exit_training:
                        break

            for epoch in range(self.nepochs):
                train_loss = self.train_epoch(model, task_id, train_loader, optimizer_main, old_model, 
                                              auxiliar_network)

                val_loss = self.validate_epoch(model, task_id, val_loader, old_model, auxiliar_network)

                test_loss, test_acc = self.test(task_id, test_loader)

                lr = self.adjust_lr(epoch, lr, optimizer_main, best_loss, patience)

                model, best_val_loss, patience, exit_training = self.adjust_lr_patience(model, optimizer_main,
                                                                                        best_val_loss,
                                                                                        val_loss, patience)
                if exit_training:
                    break



        else:
            for epoch in range(self.nepochs):
                train_loss = self.train_epoch(model, task_id, train_loader, optimizer_main)

                val_loss = self.validate_epoch(model, task_id, val_loader)

                test_loss, test_acc = self.test(task_id, test_loader)

                model, best_val_loss, patience, exit_training = self.adjust_lr_patience(model, optimizer_main,
                                                                                        best_val_loss,
                                                                                        val_loss, patience)
                if exit_training:
                    break





    def adjust_lr_patience(self, model, lr, optimizer, best_val_loss, val_loss, patience):
        exit_training = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = self.lr_patience
            best_model = copy.deepcopy(self.model)

        else:
            patience -= 1
            if patience <= 0:
                lr /= self.lr_factor
                print(' lr={:.1e}'.format(lr), end='')
                if lr < self.lr_min:
                    exit_training = True
                    print()
                patience = self.lr_patience
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                model.load_state_dict(best_model.state_dict())

        return model, lr, best_val_loss, patience, exit_training





    def train_epoch(self, model, task_id, train_loader, old_model=None, auxiliar_network=None):
        model.train()
        optimizer = self._getoptimizer()
        epoch_loss = 0

        if old_model is not None:
            old_model.eval()
            old_model.freeze_all()

        if auxiliar_network is not None:
            auxiliar_network.eval()
            auxiliar_network.freeze_all()

        for input, target in train_loader:
            input, target = self.device(input), self.device(target)
            optimizer.zero_grad()
            output = model(input)

            if old_model is not None:
                old_output = old_model(input)

            if auxiliar_network is not None:
                auxiliar_output = auxiliar_network(input)

            loss = self.criterion(output, target, task_id, old_output, auxiliar_output)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Train loss: {epoch_loss / len(train_loader)}")
        return epoch_loss / len(train_loader)

    def validate_epoch(self, model, task_id, val_loader, old_model=None, auxiliar_network=None):
        model.eval()

        loss = 0

        with torch.no_grad():
            for input, target in val_loader:
                input, target = self.device(input), self.device(target)
                output = model(input)

                if old_model is not None:
                    old_output = old_model(input)

                if auxiliar_network is not None:
                    auxiliar_output = auxiliar_network(input)

                loss += self.criterion(output, target, task_id)

        print(f"Val loss: {loss / len(val_loader)}")
        return loss / len(val_loader)
        
    def test(self, task_id, test_loader):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for input, target in test_loader:
                input, target = self.device(input), self.device(target)
                output = self.model(input)
                loss += self.criterion(output, target, task_id)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum().item()

        print(f"Test loss: {loss / len(test_loader)}")
        print(f"Test accuracy: {correct / len(test_loader.dataset)}")
        return loss / len(test_loader), correct / len(test_loader.dataset)
    
    def criterion(self, output, target, old_model_output=None, aux_output=None):
        loss = 0
        current_predictions = F.log_softmax(output, dim=1)

        if old_model_output is not None:
            
            aux_predictions = F.log_softmax(old_model_output, dim=1)

            penalty_aux = F.cross_entropy(current_predictions, aux_predictions)

            loss += self.lambda_aux * penalty_aux

        if aux_output is not None:
            
            old_predictions = F.softmax(aux_output, dim=1)

            penalty_lwf = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

            loss += self.lamb * penalty_lwf

        return loss + F.cross_entropy(output, target)





    





def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
                 aux_network: bool = False):
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
    if not aux_network:
        print(f"Train loss: {epoch_loss / len(data_loader)}")
    else:
        print(f"Auxiliar train loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)


def normal_val(model: nn.Module, data_loader: torch.utils.data.DataLoader, aux_network: bool = False):
    model.eval()
    loss = 0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = variable(input), variable(target)
            output = model(input)
            loss += F.cross_entropy(output, target)

    if not aux_network:
        print(f"Val loss: {loss / len(data_loader)}")
    else:
        print(f"Auxiliar val loss: {loss / len(data_loader)}")
    return loss / len(data_loader)


def lwf_train(model: nn.Module, old_model:nn.Module, optimizer: torch.optim, 
              data_loader: torch.utils.data.DataLoader, lwf_lambda: float):
    
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

        loss = F.cross_entropy(output, target) + lwf_lambda * penalty 

        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    print(f"Penalty: {penalty.data.item()}")
    return epoch_loss / len(data_loader)

def lwf_train_aux(model: nn.Module, old_model: nn.Module, auxiliar_network: nn.Module,
                  optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, 
                  lwf_lambda: float, lwf_aux_lambda: float):
    
    model.train()
    auxiliar_network.eval()

    epoch_loss = 0
    
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)

        # Get the predictions of the current model
        current_predictions = F.log_softmax(model(input), dim=1)
        aux_predictions = F.log_softmax(auxiliar_network(input), dim=1)
        old_predictions = F.softmax(old_model(input), dim=1)
        
        # Calculate the KL divergence between the current and old predictions
        penalty = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

        # Calculate the KL divergence between the current and aux predictions
        aux_loss = F.kl_div(current_predictions, aux_predictions, reduction='batchmean')

        # Calculate the loss of the main network
        loss = F.cross_entropy(output, target) + lwf_lambda * penalty + lwf_aux_lambda * aux_loss

        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    print(f"Penalty: {penalty.data.item()}")
    print(f"Auxiliar loss: {aux_loss.data.item()}")
    return epoch_loss / len(data_loader)




def lwf_validate(model: nn.Module, old_model:nn.Module, 
                 data_loader: torch.utils.data.DataLoader, lwf_lambda: float):
    model.eval()
    old_model.eval()
    loss = 0

    with torch.no_grad():
        for input, target in data_loader:
            input, target = variable(input), variable(target)
            output = model(input)
            
            # Get the predictions of the current model
            current_predictions = F.log_softmax(model(input), dim=1)
            old_predictions = F.softmax(old_model(input), dim=1)
            
            # Calculate the KL divergence between the current and old predictions
            penalty = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

            loss += F.cross_entropy(output, target) + lwf_lambda * penalty  
            
    print(f"Val loss: {loss / len(data_loader)}")
    return loss / len(data_loader)

def lwf_validate_aux(model: nn.Module, old_model: nn.Module, auxiliar_network: nn.Module,
                        data_loader: torch.utils.data.DataLoader, lwf_lambda: float, 
                        lwf_aux_lambda: float):
    
    model.eval()
    auxiliar_network.eval()

    loss = 0

    with torch.no_grad():
        for input, target in data_loader:
            input, target = variable(input), variable(target)
            output = model(input)

            # Get the predictions of the current model
            current_predictions = F.log_softmax(model(input), dim=1)
            aux_predictions = F.log_softmax(auxiliar_network(input), dim=1)
            old_predictions = F.softmax(old_model(input), dim=1)
            
            # Calculate the KL divergence between the current and old predictions
            penalty = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

            # Calculate the KL divergence between the current and aux predictions
            aux_loss = F.kl_div(current_predictions, aux_predictions, reduction='batchmean')

            # Calculate the loss of the main network
            loss += F.cross_entropy(output, target) + lwf_lambda * penalty + lwf_aux_lambda * aux_loss

    print(f"Val loss: {loss / len(data_loader)}")
    return loss / len(data_loader)


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
