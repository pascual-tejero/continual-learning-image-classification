import numpy as np
from urllib import request
import gzip
import pickle
import os
from os import path
from torchvision.datasets import MNIST
import requests
from tqdm import tqdm

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

def remove_not_working_mirrors_mnist():
  if not hasattr(MNIST, 'mirrors'):
    return
  
  new_mirrors = [x for x in MNIST.mirrors if "yann.lecun.com" not in x]
  if len(new_mirrors) == 0:
    return

  MNIST.mirrors = new_mirrors


def download_file_mnist(url, filename, download_path='./datasets/mnist/'):
    # Create the full path for the target directory and file
    
    target_path = os.path.join(download_path, filename)
    
    opener = request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0')
    
    # Ensure the target directory exists
    os.makedirs(download_path, exist_ok=True)
    
    # Download the file to the specified directory
    opener.retrieve(url, target_path)



def download_mnist(download_path):
    remove_not_working_mirrors_mnist()
    if not hasattr(MNIST, 'mirrors'):
      base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    else:
      base_url = MNIST.mirrors[0]
    for name in filename:
        print("Downloading " + name[1] + "...")
        download_file_mnist(base_url + name[1], name[1], download_path)
    print("Download complete.")


def save_mnist(download_path='./datasets/mnist/'):
    mnist = {}

    for name in filename[:2]:
        file_path = os.path.join(download_path, name[1])
        with gzip.open(file_path, 'rb') as f:
            tmp = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255

    for name in filename[-2:]:
        file_path = os.path.join(download_path, name[1])
        with gzip.open(file_path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open(os.path.join(download_path, 'mnist.pkl'), 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")

def init_mnist(download_path='./datasets/mnist/'):
    # Check if already downloaded:
    if path.exists(os.path.join(download_path, 'mnist.pkl')):
        print('Files already downloaded!')
    elif path.exists(os.path.join(download_path, "train-images-idx3-ubyte.gz")) and \
            path.exists(os.path.join(download_path, "t10k-images-idx3-ubyte.gz")) and \
            path.exists(os.path.join(download_path, "train-labels-idx1-ubyte.gz")) and \
            path.exists(os.path.join(download_path, "t10k-labels-idx1-ubyte.gz")):
        save_mnist(download_path)
    else:  # Download Dataset
        download_mnist(download_path)
        save_mnist(download_path)

    # MNIST(download_path, download=True)


def load_mnist(download_data='./datasets/mnist/'):
    with open(os.path.join(download_data, 'mnist.pkl'), 'rb') as f:
        mnist = pickle.load(f)
    # print(f'Train data shape: {mnist["training_images"].shape}')
    # print(f'Train labels shape: {mnist["training_labels"].shape}')
    # print(f'Test data shape: {mnist["test_images"].shape}')
    # print(f'Test labels shape: {mnist["test_labels"].shape}')
    return mnist["training_images"], mnist["training_labels"], \
           mnist["test_images"], mnist["test_labels"]


def download_fashion_mnist_file(url, file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
            progress_bar.close()

def extract_fashion_mnist(file_path, label_path):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1,28, 28)

    with gzip.open(label_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return data, labels

def load_fashion_mnist(data_dir='./datasets/fashion_mnist/'):
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    os.makedirs(data_dir, exist_ok=True)
    
    # Download and extract training data
    train_url = base_url + 'train-images-idx3-ubyte.gz'
    train_file_path = os.path.join(data_dir, 'FashionMNIST_train_images.gz')
    download_fashion_mnist_file(train_url, train_file_path)
    
    train_label_url = base_url + 'train-labels-idx1-ubyte.gz'
    train_label_file_path = os.path.join(data_dir, 'FashionMNIST_train_labels.gz')
    download_fashion_mnist_file(train_label_url, train_label_file_path)
    
    train_data, train_labels = extract_fashion_mnist(train_file_path, train_label_file_path)

    # Download and extract test data
    test_url = base_url + 't10k-images-idx3-ubyte.gz'
    test_file_path = os.path.join(data_dir, 'FashionMNIST_test_images.gz')
    download_fashion_mnist_file(test_url, test_file_path)
    
    test_label_url = base_url + 't10k-labels-idx1-ubyte.gz'
    test_label_file_path = os.path.join(data_dir, 'FashionMNIST_test_labels.gz')
    download_fashion_mnist_file(test_label_url, test_label_file_path)
    
    test_data, test_labels = extract_fashion_mnist(test_file_path, test_label_file_path)

    # Normalize greyscale values to [0, 1]
    train_data = train_data.astype(np.float32) / 255
    test_data = test_data.astype(np.float32) / 255
    
    return train_data, train_labels, test_data, test_labels
