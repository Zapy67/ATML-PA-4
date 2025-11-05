import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader

def get_cifar10(root='./data', train_transform=None, test_transform=None, batch_size=32):
    if train_transform is None:
        pass

    if test_transform is None:
        pass

    train_dataset = CIFAR10(root=root, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root=root, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

class SmallConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = nn.Sequential(
            SmallConvBlock(3, 32),
            SmallConvBlock(32, 64),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.task_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.task_head(x)

        return x

## To-Do Model Training Functions/Class

def train_model_one_epoch(model: SmallCNN, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer, device):
    pass


def evaluate_model_on_test(model: SmallCNN, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device):
    pass







