import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet model
class Net_cifar100(nn.Module):
    def __init__(self, num_classes=100):
        super(Net_cifar100, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.create_layer(16, 16, 3, stride=1)
        self.layer2 = self.create_layer(16, 32, 3, stride=2)
        self.layer3 = self.create_layer(32, 64, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def create_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def feature_extractor(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out  # Return features before the fully connected layer



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Net_cifar100(nn.Module):
#     def __init__(self):
#         super(Net_cifar100, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU()
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(128 * 4 * 4, 512)
#         self.relu4 = nn.ReLU()
#         self.fc2 = nn.Linear(512, 100)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)

#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)

#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.maxpool3(x)

#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu4(x)
#         x = self.fc2(x)

#         return x