import torch
import numpy as np

import argparse

from utils.analyse_datasets import analyse_datasets
from utils.load_mnist import init_mnist, load_mnist, load_fashion_mnist

# import mnist_utils as mnist_utils

def get_dataset_mnist(args):
    """
    In this function, we get the datasets for training and testing.
    """


    # Load the MNIST dataset
    init_mnist()
    train_mdata, train_mlabels_data, test_mdata, test_mlabels = load_mnist()

    # Load the Fashion MNIST dataset
    train_fdata, train_flabels_data, test_fdata, test_flabels = load_fashion_mnist()


    # Split train_images and train_labels into training and validation for MNIST dataset
    train_mimages = train_mdata[:59900]
    train_mlabels = train_mlabels_data[:59900]
    val_mimages = train_mdata[59900:]
    val_mlabels = train_mlabels_data[59900:]


    # Split train_images and train_labels into training and validation for fashion MNIST dataset
    train_fimages = train_fdata[:59900]
    train_flabels = train_flabels_data[:59900]
    val_fimages = train_fdata[59900:]
    val_flabels = train_flabels_data[59900:]

    # Transform the images to tensors. Converting a tensor from a list is extremely slow, so we use numpy arrays
    train_images_1 = torch.tensor(np.array(train_mimages), dtype=torch.float32)
    train_labels_1 = torch.tensor(np.array(train_mlabels), dtype=torch.int64)
    train_images_2 = torch.tensor(np.array(train_fimages), dtype=torch.float32)
    train_labels_2 = torch.tensor(np.array(train_flabels), dtype=torch.int64)

    val_images_1 = torch.tensor(np.array(val_mimages), dtype=torch.float32)
    val_labels_1 = torch.tensor(np.array(val_mlabels), dtype=torch.int64)
    val_images_2 = torch.tensor(np.array(val_fimages), dtype=torch.float32)
    val_labels_2 = torch.tensor(np.array(val_flabels), dtype=torch.int64)

    test_images_1 = torch.tensor(np.array(test_mdata), dtype=torch.float32)
    test_labels_1 = torch.tensor(np.array(test_mlabels), dtype=torch.int64)
    test_images_2 = torch.tensor(np.array(test_fdata), dtype=torch.float32)
    test_labels_2 = torch.tensor(np.array(test_flabels), dtype=torch.int64)

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


# if __name__ == '__main__':
#     argparse = argparse.ArgumentParser()
#     argparse.add_argument('--dataset', type=str, default='mnist')
#     get_dataset_mnist(argparse.parse_args())
