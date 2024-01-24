import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x
    
    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        out_horizontal = torch.mean(x, dim=3)  # Average pooling along dimension 3 (width)
        out_vertical = torch.mean(x, dim=2) # Average pooling along dimension 2 (height)

        # Concatenate the two outputs
        out_pooled = torch.cat([out_horizontal, out_vertical], dim=2).view(x.size(0), -1).view(x.size(0), -1)

        return out_pooled


# class Net_cifar10(nn.Module):
#     """
#     Neural network with 2 convolutional layers and 2 fully connected layers with dropout.

#     This neural network is used for the CIFAR-10 dataset. The input size is 32x32. Also, it is going to be used
#     for testing methods about continual learning.

#     The neural network is defined as follows:
#         - 2 convolutional layers with kernel size of 5 and stride of 1
#         - 2 max pooling layers with kernel size of 2 and stride of 2
#         - 2 fully connected layers with dropout
#         - 1 output layer (fully connected layer)

#     The neural network is defined as follows:
    
#             Input -> Convolutional layer -> Max pooling layer -> Convolutional layer -> Max pooling layer -> 
#             Fully connected layer -> Dropout layer -> Fully connected layer -> Output 
#     """

#     def __init__(self):
#         super(Net_cifar10, self).__init__()
#         # Define the layers
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5) # 3 input channel, 10 output channels, kernel size of 5
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 10 input channels, 20 output channels, kernel size of 5
#         self.conv2_drop = nn.Dropout2d() # Dropout layer
#         self.fc1 = nn.Linear(500, 50) # 500 input features, 50 output features
#         self.fc2 = nn.Linear(50, 10) # 50 input features, 10 output features

#     def forward(self, x):
#         # Define the forward pass
#         x = F.relu(F.max_pool2d(self.conv1(x), 2)) # Max pooling over a (2, 2) window
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # Max pooling over a (2, 2) window
#         x = x.view(-1, 500) # Reshape the tensor
#         x = F.relu(self.fc1(x)) # Apply the ReLU activation function
#         x = F.dropout(x, training=self.training) # Apply the dropout layer
#         x = self.fc2(x) # Apply the linear layer
#         return x
#         # return F.log_softmax(x, dim=1) # Apply the log softmax function