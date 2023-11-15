import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_mnist(nn.Module):
    """
    Neural network with 2 convolutional layers and 2 fully connected layers with dropout.

    This neural network is used for the MNIST dataset. The input size is 28x28. Also, it is going to be used
    for testing methods about continual learning.

    The neural network is defined as follows:
        - 2 convolutional layers with kernel size of 5 and stride of 1
        - 2 max pooling layers with kernel size of 2 and stride of 2
        - 2 fully connected layers with dropout
        - 1 output layer (fully connected layer)

    The neural network is defined as follows:
    
            Input -> Convolutional layer -> Max pooling layer -> Convolutional layer -> Max pooling layer -> 
            Fully connected layer -> Dropout layer -> Fully connected layer -> Output 
    """

    def __init__(self):
        super(Net_mnist, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1 input channel, 10 output channels, kernel size of 5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 10 input channels, 20 output channels, kernel size of 5
        self.conv2_drop = nn.Dropout2d() # Dropout layer
        self.fc1 = nn.Linear(320, 50) # 320 input features, 50 output features
        self.fc2 = nn.Linear(50, 10) # 50 input features, 10 output features

    def forward(self, x):
        # Define the forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # Max pooling over a (2, 2) window
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # Max pooling over a (2, 2) window
        x = x.view(-1, 320) # Reshape the tensor
        x = F.relu(self.fc1(x)) # Apply the ReLU activation function
        x = F.dropout(x, training=self.training) # Apply the dropout layer
        x = self.fc2(x) # Apply the linear layer
        return x
        # return F.log_softmax(x, dim=1) # Apply the log softmax function