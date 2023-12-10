import os
import torch

def save_model(model, args, id_task_dataset, task="naive", joint_training=False):
    """
    This function saves the model in the path: "./models/models_saved/{args.dataset}/{task}_training/
    model_{args.dataset}_{id_task_dataset}.pt"

    :param model: model to save
    :param args: arguments from the command line
    :param id_task_dataset: id of the task or dataset
    :param task: task name
    :param joint_training: boolean to indicate if the model is going to be saved in the joint training folder
    :return: None
    """

    os.makedirs(f'./models/models_saved/{args.exp_name}/{task}_training_{args.dataset}', exist_ok=True)
    path = f'./models/models_saved/{args.exp_name}/{task}_training_{args.dataset}'

    if not joint_training:
        torch.save(model.state_dict(), f'{path}/model_{task}_aftertask_{id_task_dataset}_{args.dataset}.pt')
    else:
        torch.save(model.state_dict(), f'{path}/model_{task}_{args.dataset}.pt')