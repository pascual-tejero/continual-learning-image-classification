import os
import torch

def save_model(model, args, id_task_dataset, method="naive", joint_datasets=False):
    """
    This function saves the model in the path: "./models/models_saved/{args.dataset}/{method}_training/
    model_{args.dataset}_{id_task_dataset}.pt"

    :param model: model to save
    :param args: arguments from the command line
    :param id_task_dataset: id of the method or dataset
    :param method: method name
    :param joint_datasets: boolean to indicate if the model is going to be saved in the joint training folder
    :return: None
    """

    os.makedirs(f'./models/models_saved/{args.exp_name}/{method}_{args.dataset}', exist_ok=True)
    path = f'./models/models_saved/{args.exp_name}/{method}_{args.dataset}'

    if not joint_datasets:
        torch.save(model.state_dict(), f'{path}/{method}_aftertask_{id_task_dataset}_{args.dataset}.pt')
    else:
        torch.save(model.state_dict(), f'{path}/{method}_{args.dataset}.pt')