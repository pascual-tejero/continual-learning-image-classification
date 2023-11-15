import numpy as np
from urllib import request
import gzip
import pickle
import os
from os import path
from torchvision.datasets import MNIST

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

def remove_not_working_mirrors():
  if not hasattr(MNIST, 'mirrors'):
    return
  
  new_mirrors = [x for x in MNIST.mirrors if "yann.lecun.com" not in x]
  if len(new_mirrors) == 0:
    return

  MNIST.mirrors = new_mirrors


def download_file(url, filename):
    # Create the full path for the target directory and file
    target_directory = "./datasets/mnist"
    target_path = os.path.join(target_directory, filename)
    
    opener = request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0')
    
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)
    
    # Download the file to the specified directory
    opener.retrieve(url, target_path)



def download_mnist():
    remove_not_working_mirrors()
    if not hasattr(MNIST, 'mirrors'):
      base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    else:
      base_url = MNIST.mirrors[0]
    for name in filename:
        print("Downloading " + name[1] + "...")
        download_file(base_url + name[1], name[1])
    print("Download complete.")


# def save_mnist():
#     mnist = {}
#     for name in filename[:2]:
#         with gzip.open(name[1], 'rb') as f:
#             tmp = np.frombuffer(f.read(), np.uint8, offset=16)
#             mnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255
#     for name in filename[-2:]:
#         with gzip.open(name[1], 'rb') as f:
#             mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
#     with open("mnist.pkl", 'wb') as f:
#         pickle.dump(mnist, f)
#     print("Save complete.")

def save_mnist():
    mnist = {}
    data_dir = "./datasets/mnist/"  # Path to the directory where the MNIST dataset files are located

    for name in filename[:2]:
        file_path = os.path.join(data_dir, name[1])
        with gzip.open(file_path, 'rb') as f:
            tmp = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255

    for name in filename[-2:]:
        file_path = os.path.join(data_dir, name[1])
        with gzip.open(file_path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open("./datasets/mnist/mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")

def init():
    # Check if already downloaded:
    if path.exists("./datasets/mnist/mnist.pkl"):
        print('Files already downloaded!')
    elif path.exists("./datasets/mnist/train-images-idx3-ubyte.gz") and \
            path.exists("./datasets/mnist/t10k-images-idx3-ubyte.gz") and \
            path.exists("./datasets/mnist/train-labels-idx1-ubyte.gz") and \
            path.exists("./datasets/mnist/t10k-labels-idx1-ubyte.gz"):
        save_mnist()
    else:  # Download Dataset
        download_mnist()
        save_mnist()

    MNIST(path.join('data', 'mnist'), download=True)


def load():
    with open("./datasets/mnist/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    # print(f'Train data shape: {mnist["training_images"].shape}')
    # print(f'Train labels shape: {mnist["training_labels"].shape}')
    # print(f'Test data shape: {mnist["test_images"].shape}')
    # print(f'Test labels shape: {mnist["test_labels"].shape}')
    return mnist["training_images"], mnist["training_labels"], \
           mnist["test_images"], mnist["test_labels"]


if __name__ == '__main__':
    init()
