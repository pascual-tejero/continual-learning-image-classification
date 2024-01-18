import copy

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


  


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
                 loss_ANCL=None):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        if loss_ANCL is None:
            loss = F.cross_entropy(output, target)
        else:
            loss = criterion(output, target, task=0)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)


def normal_val(model: nn.Module, data_loader: torch.utils.data.DataLoader, loss_ANCL=None):
    model.eval()
    loss = 0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = variable(input), variable(target)
            output = model(input)
            if loss_ANCL is None:
                loss += F.cross_entropy(output, target)
            else:
                loss += criterion(output, target, task=0)

    print(f"Val loss: {loss / len(data_loader)}")
    return loss / len(data_loader)


def lwf_train(model: nn.Module, old_model:nn.Module, optimizer: torch.optim, 
              data_loader: torch.utils.data.DataLoader, alpha: float, loss_ANCL=None):
    model.train()
    epoch_loss = 0
    epoch_loss = 0
    epoch_penalty_loss = 0

    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)

        # Get the predictions of the current model
        current_predictions = F.log_softmax(model(input), dim=1)
        old_predictions = F.softmax(old_model(input), dim=1)
        
        # Calculate the KL divergence between the current and old predictions
        penalty = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

        if loss_ANCL is None:
            loss = F.cross_entropy(output, target) + alpha * penalty
        else:
            old_pred = old_model(input)
            loss = criterion(output, target, task=1, targets_old=old_pred, lwf_lambda=alpha)

        epoch_loss += loss.data.item()
        epoch_penalty_loss += penalty.data.item() * alpha
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    print(f"Penalty: {epoch_penalty_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)


def lwf_validate(model: nn.Module, old_model:nn.Module, data_loader: torch.utils.data.DataLoader, 
                 alpha: float, loss_ANCL=None):
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

            if loss_ANCL is None:
                loss += F.cross_entropy(output, target) + alpha * penalty
            else:
                old_pred = old_model(input)
                loss += criterion(output, target, task=1, targets_old=old_pred, lwf_lambda=alpha)
            
    print(f"Val loss: {loss / len(data_loader)}")
    return loss / len(data_loader)

def lwf_train_aux(model, old_model, optimizer, data_loader, lwf_lambda, auxiliary_network, lwf_aux_lambda,
                  loss_ANCL=None):
    model.train()
    auxiliary_network.eval()
    epoch_loss = 0
    epoch_penalty_loss = 0
    epoch_aux_loss = 0


    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)

        # Get the predictions of the current model
        current_predictions = F.log_softmax(model(input), dim=1)
        old_predictions = F.softmax(old_model(input), dim=1)

        aux_pred = auxiliary_network(input)
        
        # Calculate the KL divergence between the current and old predictions
        penalty = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

        # Get the predictions of the auxiliary network
        aux_loss = F.cross_entropy(auxiliary_network(input), target)

        if loss_ANCL is None:
            loss = F.cross_entropy(output, target) + lwf_lambda * penalty + lwf_aux_lambda * aux_loss
        else:
            old_pred = old_model(input)
            loss = criterion(output, target, task=1, targets_old=old_pred, lwf_lambda=lwf_lambda,
                            targets_aux=aux_pred, lwf_aux_lambda=lwf_aux_lambda)

        epoch_loss += loss.data.item()
        epoch_penalty_loss += penalty.data.item() * lwf_lambda
        epoch_aux_loss += aux_loss.data.item() * lwf_aux_lambda
        loss.backward()
        optimizer.step()

    print(f"Train loss: {epoch_loss / len(data_loader)}")
    print(f"Penalty: {epoch_penalty_loss / len(data_loader)}")
    print(f"Auxiliar loss: {epoch_aux_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)

def lwf_validate_aux(model, old_model, data_loader, lwf_lambda, auxiliary_network, lwf_aux_lambda,
                     loss_ANCL=None):
    model.eval()
    old_model.eval()
    auxiliary_network.eval()
    loss = 0

    with torch.no_grad():
        for input, target in data_loader:
            input, target = variable(input), variable(target)
            output = model(input)
            
            # Get the predictions of the current model
            current_predictions = F.log_softmax(model(input), dim=1)
            old_predictions = F.softmax(old_model(input), dim=1)
            aux_pred = auxiliary_network(input)
            
            # Calculate the KL divergence between the current and old predictions
            penalty = F.kl_div(current_predictions, old_predictions, reduction='batchmean')

            # Get the predictions of the auxiliary network
            aux_loss = F.cross_entropy(auxiliary_network(input), target)

            if loss_ANCL is None:
                loss += F.cross_entropy(output, target) + lwf_lambda * penalty + lwf_aux_lambda * aux_loss
            else:
                old_pred = old_model(input)
                loss += criterion(output, target, task=1, targets_old=old_pred, lwf_lambda=lwf_lambda,
                                targets_aux=aux_pred, lwf_aux_lambda=lwf_aux_lambda)

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

        test_task_list.append(id_task_test+1)
        test_loss_list.append(test_loss)
        test_acc_list.append(accuracy)

        print(f"Test on task {id_task_test+1}: Average loss: {test_loss:.6f}, "
              f"Accuracy: {accuracy:.2f}%")

    avg_acc /= len(datasets)
    print(f"Average accuracy: {avg_acc:.2f}%")

    return test_task_list, test_loss_list, test_acc_list, avg_acc


def criterion(outputs, targets, task=0, targets_old=None, lwf_lambda=None, targets_aux=None, lwf_aux_lambda=None):
    "Return the loss value"
    T = 2.0
    loss = 0
    if task > 0:
        loss += lwf_lambda * cross_entropy(outputs, targets_old, exp=1.0/T)

        if targets_aux is not None:
            loss += lwf_aux_lambda * cross_entropy(outputs, targets_aux, exp=1.0/T)

    return loss + F.cross_entropy(outputs, targets)

def cross_entropy(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    # print(outputs.shape)
    # print(targets.shape)
    out = torch.nn.functional.softmax(outputs, dim=1)
    tar = torch.nn.functional.softmax(targets, dim=1)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
        # print(ce.shape)
    return ce    
    

# def adjust_lr_patience(val_loss_epoch, best_val_loss, patience, lr, args, model, model_best, optimizer,
#                        test_acc_final, test_tasks_accuracy, avg_accuracy, epoch):
#     # Early stopping
#     if val_loss_epoch < best_val_loss:
#         best_val_loss = val_loss_epoch
#         patience = args.lr_patience
#         model_best = copy.deepcopy(model)
#         return best_val_loss, patience, model_best
#     else:
#         # if the loss does not go down, decrease patience
#         patience -= 1
#         if patience <= 0:
#             # if it runs out of patience, reduce the learning rate
#             lr /= args.lr_decay
#             print(' lr={:.1e}'.format(lr), end='')
#             if lr < args.lr_min:
#                 # if the lr decreases below minimum, stop the training session
#                 print()
#                 test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 
#                 return best_val_loss, patience, model_best
#             # reset patience and recover best model so far to continue training
#             patience = args.lr_patience
#             optimizer.param_groups[0]['lr'] = lr
#             model.load_state_dict(model_best.state_dict())

#     # Save the results of the epoch if it is the last epoch
#     if epoch == args.epochs-1:
#         test_acc_final.append([test_tasks_accuracy, avg_accuracy]) 
#     return best_val_loss, patience, model_best