import os
import numpy as np
import urllib.request
import tarfile
from six.moves import cPickle as pickle
import platform

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def download_CIFAR10(download_path='./datasets/cifar10/'):
    """Download and extract the CIFAR-10 dataset."""
    dataset_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_name = 'cifar-10-python.tar.gz'
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    out_path = os.path.join(download_path, file_name)
    
    # Download the file using urllib.request
    print('Downloading CIFAR-10 from {}'.format(dataset_link))
    urllib.request.urlretrieve(dataset_link, out_path)
    
    # Extract the tar.gz file using tarfile
    print('Extracting CIFAR-10 from {}'.format(out_path))
    with tarfile.open(out_path, 'r:gz') as tar:
        tar.extractall(download_path)
    
    print('Done!')

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000, download_path='./datasets/cifar10/'):


    os.makedirs(download_path, exist_ok=True)

    # Download the raw CIFAR-10 data
    if os.path.exists(os.path.join(download_path, 'cifar-10-batches-py')) == False:
        download_CIFAR10(download_path)

    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(download_path, 'cifar-10-batches-py')
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    x_train = x_train.astype('float64')/np.max(x_train)
    x_val = x_val.astype('float64')/np.max(x_val)
    x_test = x_test.astype('float64')/np.max(x_test)

    x_train = x_train.reshape(-1, 3, img_rows, img_cols)
    x_val = x_val.reshape(-1, 3, img_rows, img_cols)
    x_test = x_test.reshape(-1, 3, img_rows, img_cols)

    return x_train, y_train, x_val, y_val, x_test, y_test


# Invoke the above function to get our data.
# x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()


