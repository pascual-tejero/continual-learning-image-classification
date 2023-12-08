import torch
import numpy as np

from sklearn.datasets import fetch_openml
import argparse

def analyse_datasets(datasets, args):
    """
    Count number of images per class in the dataset.
    """
    print("=" * 130)
    print("Number of tasks: ", len(datasets))
    print("Number of samples per task: ", [len(task[0]) for task in datasets])
    print("=" * 130)

    # Load the dataset
    for idx, (train_set, val_set, test_set) in enumerate(datasets):

        # Split each tensor dataset into images and labels
        x_train, t_train = train_set.tensors
        x_val, t_val = val_set.tensors
        x_test, t_test = test_set.tensors

        # Convert the tensors to numpy arrays
        x_train = x_train.numpy()
        t_train = t_train.numpy()
        x_val = x_val.numpy()
        t_val = t_val.numpy()
        x_test = x_test.numpy()
        t_test = t_test.numpy()
        

        # Get the number of images per class
        count_train = {}
        count_val = {}
        count_test = {}

        # Count the number of images per class in the sets
        for i in range(len(t_train)):
            if t_train[i] in count_train:
                count_train[t_train[i]] += 1
            else:
                count_train[t_train[i]] = 1

        for i in range(len(t_val)):
            if t_val[i] in count_val:
                count_val[t_val[i]] += 1
            else:
                count_val[t_val[i]] = 1

        for i in range(len(t_test)):
            if t_test[i] in count_test:
                count_test[t_test[i]] += 1
            else:
                count_test[t_test[i]] = 1

        # Sort dictionary by key
        count_train = dict(sorted(count_train.items()))
        count_val = dict(sorted(count_val.items()))
        count_test = dict(sorted(count_test.items()))

        # Print the results
        print(f"Train set {idx+1}: {count_train}")
        print(f"Val set {idx+1}: {count_val}")
        print(f"Test set {idx+1}: {count_test}")
        print("=" * 130)
    


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset', type=str, default='mnist')
    analyse_datasets(argparse.parse_args())
