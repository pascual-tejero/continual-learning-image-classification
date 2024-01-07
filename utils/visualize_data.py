import torch
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import torchvision
import random

# Go to one directory below and then add the current path to the sys.path
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from load_mnist import init_mnist, load_mnist, load_fashion_mnist
from load_cifar10 import get_CIFAR10_data
from load_cifar100 import get_CIFAR100_data

def visualize_data(save_image = False, args=None):
    """
    In this function, we get the datasets for training and testing.
    """

    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(args)

    # Get the number of classes
    num_classes = len(np.unique(y_train))

    # Visualize the data
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        visualize_mnist(x_train, y_train, num_classes, save_image)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        visualize_cifar(x_train, y_train, num_classes, save_image)
    else:
        raise ValueError('Dataset not found!')


def load_dataset(args):
    if args.dataset == 'mnist':
        download_path = '../datasets/mnist/'

        # Load the MNIST dataset
        init_mnist(download_path)
        x_train, y_train,  x_test, y_test = load_mnist(download_path)

        x_train = x_train[:49000]
        y_train = y_train[:49000]
        x_val = x_train[49000:]
        y_val = y_train[49000:]

    elif args.dataset == 'fashion_mnist':
        # Download path for the Fashion MNIST dataset
        download_path = '../datasets/fashion_mnist/' 

        # Load the Fashion MNIST dataset
        x_train, y_train, x_test, y_test = load_fashion_mnist(download_path)

        x_train = x_train[:49000]
        y_train = y_train[:49000]
        x_val = x_train[49000:]
        y_val = y_train[49000:]

    elif args.dataset == 'cifar10':
        # Download path for the CIFAR10 dataset
        download_path = '../datasets/cifar10/'

        # Get the images and labels from the dataset
        x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data(num_training=49000, 
                                                                          num_validation=1000, 
                                                                          num_test=10000, 
                                                                          download_path=download_path)


    elif args.dataset == 'cifar100':
        download_path = '../datasets/cifar100/'

        # Load the CIFAR100 dataset
        x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR100_data(num_training=49000, 
                                                                          num_validation=1000, 
                                                                          num_test=10000, 
                                                                          download_path=download_path)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def visualize_mnist(x, y, num_classes, save_image=False):
    # Define the labels dictionary
    if args.dataset == 'mnist':
        labels_dict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
                       5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
    elif args.dataset == 'fashion_mnist':
        labels_dict = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                       5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
        
    # Random number to visualize
    random_numbers = [random.randint(0, len(x)) for i in range(num_classes)]
    

    # Visualize the data
    for idx, i in enumerate(random_numbers):
        plt.imshow(x[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {labels_dict[y[i]]} ({y[i]})")
        plt.axis('off')
        plt.show()
        fig = plt.figure()

        if save_image:
            fig.savefig(f"mnist_{idx}.png")


def visualize_cifar(x, y, num_classes, save_image=False):
    # Define the labels dictionary
    if args.dataset == 'cifar10':
        labels_dict = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
    elif args.dataset == 'cifar100':
        labels_dict = {
            0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 
            8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 
            15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 
            22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'crab', 27: 'crocodile', 28: 'cup', 
            29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 
            36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'computer_keyboard', 40: 'lamp', 41: 'lawn_mower', 
            42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 
            49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 
            56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy', 
            63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 
            70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 
            78: 'snake', 79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 
            84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 
            90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 
            97: 'wolf', 98: 'woman', 99: 'worm'
        }


    # Random number to visualize
    random_numbers = [random.randint(0, len(x)) for i in range(num_classes)]

    # Visualize the data
    for idx, i in enumerate(random_numbers):
        plt.imshow(np.moveaxis(x[i], 0, -1))
        plt.title(f"Label: {labels_dict[y[i]]} ({y[i]})")
        plt.axis('off')
        plt.show()
        fig = plt.figure()

        if save_image:
            fig.savefig(f"mnist_{idx}.png")

if __name__ == "__main__":
    # Define the arguments
    parser = argparse.ArgumentParser(description='Continual learning')

    # Datasets: mnist, fashion_mnist, cifar10, cifar100
    parser.add_argument('--dataset', type=str, default='cifar100', metavar='N',
                        help='dataset to use (default: mnist)')

    args = parser.parse_args()

    visualize_data(save_image=False, args=args)