import torch
import numpy as np

import argparse

from utils.analyse_datasets import analyse_datasets
import utils.mnist_utils as mnist_utils
import utils.cifar10_utils as cifar10_utils

# import mnist_utils as mnist_utils

def get_datasets(args):
    """
    In this function, we get the datasets for training and testing.
    """

    if args.dataset == 'mnist':
        mnist_utils.init()

        # x_train, t_train, x_test, t_test = mnist.load()
        train_images, train_labels, test_images, test_labels = mnist_utils.load()

        # Split train_images and train_labels into training and validation
        val_images = train_images[55000:]
        val_labels = train_labels[55000:]
        train_images = train_images[:55000]
        train_labels = train_labels[:55000]



    elif args.dataset == 'cifar10':
        # Get the images and labels from the dataset
        train_images, train_labels, val_images, val_labels, test_images, test_labels = cifar10_utils.get_CIFAR10_data(
                                                                                        num_training=49000, 
                                                                                        num_validation=1000, 
                                                                                        num_test=10000
                                                                                        )
    print('Train data shape: ', train_images.shape)
    print('Train labels shape: ', train_labels.shape)
    print('Validation data shape: ', val_images.shape)
    print('Validation labels shape: ', val_labels.shape)
    print('Test data shape: ', test_images.shape)
    print('Test labels shape: ', test_labels.shape)
        
    # First and second dataset
    train_images_1, train_labels_1 = [], []
    val_images_1, val_labels_1 = [], [] 
    test_images_1, test_labels_1 = [], []

    train_images_2, train_labels_2 = [], []
    val_images_2, val_labels_2 = [], []
    test_images_2, test_labels_2 = [], []

    # count_3, count_6, count_8 = 0, 0, 0 # Count the number of images from 3, 6 and 8

    # Get the images and labels from the first and second dataset for the training dataset
    for i in range(len(train_images)):
        if (train_labels[i] == 0 or train_labels[i] == 2 or train_labels[i] == 4 or 
            train_labels[i] == 6 or train_labels[i] == 8):
            train_images_1.append(train_images[i])
            train_labels_1.append(train_labels[i])
            # if count_3 < 100 and train_labels[i] == 3:
            #     train_images_2.append(train_images[i])
            #     train_labels_2.append(train_labels[i])
            #     count_3 += 1
            # elif count_6 < 100 and train_labels[i] == 6:
            #     train_images_2.append(train_images[i])
            #     train_labels_2.append(train_labels[i])
            #     count_6 += 1
            # elif count_8 < 100 and train_labels[i] == 8:
            #     train_images_2.append(train_images[i])
            #     train_labels_2.append(train_labels[i])
            #     count_8 += 1
            # else:
            #     train_images_1.append(train_images[i])
            #     train_labels_1.append(train_labels[i])
        else:
            train_images_2.append(train_images[i])
            train_labels_2.append(train_labels[i])

    # Get the images and labels from the first and second dataset for the validation dataset
    for i in range(len(val_images)):
        if (val_labels[i] == 0 or val_labels[i] == 2 or val_labels[i] == 4 or 
            val_labels[i] == 6 or val_labels[i] == 8):
            # If the label is 3, 6 or 8, then add the image and label to the first dataset
            val_images_1.append(val_images[i])
            val_labels_1.append(val_labels[i])
        else:
            # If the label is not 3, 6 or 8, then add the image and label to the first dataset
            val_images_2.append(val_images[i])
            val_labels_2.append(val_labels[i])

    # Get the images and labels from the first and second dataset for the test dataset
    for i in range(len(test_images)):
        if (test_labels[i] == 0 or test_labels[i] == 2 or test_labels[i] == 4 or 
            test_labels[i] == 6 or test_labels[i] == 8):
            # If the label is 3, 6 or 8, then add the image and label to the first dataset
            test_images_1.append(test_images[i])
            test_labels_1.append(test_labels[i])
        else:
            # If the label is not 3, 6 or 8, then add the image and label to the first dataset
            test_images_2.append(test_images[i])
            test_labels_2.append(test_labels[i])
            
    
    

    # Transform the images to tensors. Converting a tensor from a list is extremely slow, so we use numpy arrays
    train_images_1 = torch.tensor(np.array(train_images_1), dtype=torch.float32)
    train_labels_1 = torch.tensor(np.array(train_labels_1), dtype=torch.int64)
    train_images_2 = torch.tensor(np.array(train_images_2), dtype=torch.float32)
    train_labels_2 = torch.tensor(np.array(train_labels_2), dtype=torch.int64)

    val_images_1 = torch.tensor(np.array(val_images_1), dtype=torch.float32)
    val_labels_1 = torch.tensor(np.array(val_labels_1), dtype=torch.int64)
    val_images_2 = torch.tensor(np.array(val_images_2), dtype=torch.float32)
    val_labels_2 = torch.tensor(np.array(val_labels_2), dtype=torch.int64)

    test_images_1 = torch.tensor(np.array(test_images_1), dtype=torch.float32)
    test_labels_1 = torch.tensor(np.array(test_labels_1), dtype=torch.int64)
    test_images_2 = torch.tensor(np.array(test_images_2), dtype=torch.float32)
    test_labels_2 = torch.tensor(np.array(test_labels_2), dtype=torch.int64)

    # Create the datasets
    train_dataset_1 = torch.utils.data.TensorDataset(train_images_1, train_labels_1)
    val_dataset_1 = torch.utils.data.TensorDataset(val_images_1, val_labels_1)
    test_dataset_1 = torch.utils.data.TensorDataset(test_images_1, test_labels_1)

    train_dataset_2 = torch.utils.data.TensorDataset(train_images_2, train_labels_2)
    val_dataset_2 = torch.utils.data.TensorDataset(val_images_2, val_labels_2)
    test_dataset_2 = torch.utils.data.TensorDataset(test_images_2, test_labels_2)

    # Join the datasets
    datasets = [(train_dataset_1, val_dataset_1, test_dataset_1), (train_dataset_2, val_dataset_2, test_dataset_2)]

    # Analyse the datasets
    analyse_datasets(datasets, args)

    return datasets


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset', type=str, default='mnist')
    get_datasets(argparse.parse_args())
