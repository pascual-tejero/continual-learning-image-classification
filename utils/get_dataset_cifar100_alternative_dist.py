import numpy as np
import torch

from utils.analyse_datasets import analyse_datasets
from utils.load_cifar100 import get_CIFAR100_data



def get_dataset_cifar100_alternative_dist(args):
    """
    In this function, we get the datasets for training and testing.
    """

    # Load the CIFAR100 dataset
    x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR100_data()

    num_tasks = args.num_tasks
    num_classes = 100

    # Create the tasks
    datasets_tasks = create_tasks_alternative_dist(x_train, y_train, x_val, y_val, x_test, y_test, num_tasks, num_classes, args)

    # Create the dataset for joint training
    # dataset_joint_training = create_joint_training_dataset(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, args)

    return datasets_tasks


def create_tasks_alternative_dist(x_train, y_train, x_val, y_val, x_test, y_test, num_tasks, num_classes, args):

    # Divide the dataset into 2 tasks: 80% of the classes in the first task, 20% in the second task
    # Also, 5% of the classes of the second task are also in the first task
    list_tasks = [80, 100]
    tasks_dict = {}
    for i, value in enumerate(list_tasks):
        previous_value = list_tasks[i-1] if i > 0 else 0
        tasks_dict[i] = list(range(previous_value, value))

    # # Add the 5% of the classes of the second task that are also in the first task
    # tasks_dict[0].extend(tasks_dict[1][:1])

    datasets_tasks = [] # [train, val, test]

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []

    # Split dataset into tasks based on the tasks_dict
    for key, value in tasks_dict.items():

        train_task_images = []
        train_task_labels = []
        val_task_images = []
        val_task_labels = []
        test_task_images = []
        test_task_labels = []

        # Get the images and labels of the task
        for i, label in enumerate(y_train):
            if label in value:
                train_task_images.append(x_train[i])
                train_task_labels.append(label)
        for i, label in enumerate(y_val):
            if label in value:
                val_task_images.append(x_val[i])
                val_task_labels.append(label)
        for i, label in enumerate(y_test):
            if label in value:
                test_task_images.append(x_test[i])
                test_task_labels.append(label)

        # Add the images and labels to the final list
        train_images.append(train_task_images)
        train_labels.append(train_task_labels)
        val_images.append(val_task_images)
        val_labels.append(val_task_labels)
        test_images.append(test_task_images)
        test_labels.append(test_task_labels)

    # Add 5% of the data (randomly) of each class of the second task to the first task
    for i in tasks_dict[1]:
        # Get the indexes of the images of the second task that have the same label as the current image
        indexes = [j for j, x in enumerate(y_train) if x == i]
        # Get the number of images to add to the first task
        num_images_to_add = int(len(indexes) * 0.05)
        # Get the indexes of the images to add to the first task
        indexes_to_add = np.random.choice(indexes, num_images_to_add, replace=False)
        # Add the images and labels to the first task
        train_images[0].extend([x_train[j] for j in indexes_to_add])
        train_labels[0].extend([y_train[j] for j in indexes_to_add])

    for i in range(len(train_images)):
        train_images_task = train_images[i]
        train_labels_task = train_labels[i]
        val_images_task = val_images[i]
        val_labels_task = val_labels[i]
        test_images_task = test_images[i]
        test_labels_task = test_labels[i]
     
        # Transform the images to tensors. Converting a tensor from a list is extremely slow, so we use numpy arrays
        train_images_data = torch.tensor(np.array(train_images_task), dtype=torch.float32)
        train_labels_data = torch.tensor(np.array(train_labels_task), dtype=torch.int64)
        val_images_data = torch.tensor(np.array(val_images_task), dtype=torch.float32)
        val_labels_data = torch.tensor(np.array(val_labels_task), dtype=torch.int64)
        test_images_data = torch.tensor(np.array(test_images_task), dtype=torch.float32)
        test_labels_data = torch.tensor(np.array(test_labels_task), dtype=torch.int64)

        # Create the datasets
        train_dataset = torch.utils.data.TensorDataset(train_images_data, train_labels_data)
        val_dataset = torch.utils.data.TensorDataset(val_images_data, val_labels_data)
        test_dataset = torch.utils.data.TensorDataset(test_images_data, test_labels_data)

        datasets_tasks.append([train_dataset, val_dataset, test_dataset]) # [train, val, test]
     
    analyse_datasets(datasets_tasks, args)

    return datasets_tasks

