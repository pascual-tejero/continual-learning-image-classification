# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Basic block for ResNet
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# # ResNet model
# class Net_cifar100(nn.Module):
#     def __init__(self, num_classes=100):
#         super(Net_cifar100, self).__init__()
#         self.in_channels = 16
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self.create_layer(16, 16, 3, stride=1)
#         self.layer2 = self.create_layer(16, 32, 3, stride=2)
#         self.layer3 = self.create_layer(32, 64, 3, stride=2)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(64, num_classes)

#     def create_layer(self, in_channels, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(BasicBlock(in_channels, out_channels, stride))
#             in_channels = out_channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

#     def feature_extractor(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)

#         horizontal_pool = torch.mean(out, dim=3)  # Average pooling along dimension 3 (width)
#         vertical_pool = torch.mean(out, dim=2) # Average pooling along dimension 2 (height)

#         # Concatenate the two outputs
#         pooled_features = torch.cat([horizontal_pool, vertical_pool], dim=2).view(out.size(0), -1) 
#         return pooled_features


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
        self.layer1 = self.create_layer(16, 32, 3, stride=1)
        self.layer2 = self.create_layer(32, 64, 3, stride=2)
        self.layer3 = self.create_layer(64, 128, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

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

        horizontal_pool = torch.mean(out, dim=3)  # Average pooling along dimension 3 (width)
        vertical_pool = torch.mean(out, dim=2)  # Average pooling along dimension 2 (height)

        # Concatenate the two outputs
        pooled_features = torch.cat([horizontal_pool, vertical_pool], dim=2).view(out.size(0), -1)
        return pooled_features