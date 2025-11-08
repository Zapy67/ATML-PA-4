"""
utils.py
====================
This module provides utility components for Federated Learning (FL) simulations,
including model definitions, training/evaluation routines, and dataset partitioning
for both homogeneous and (future) heterogeneous domain splits.

Main Components:
----------------
1. **Model Definitions**
   - `SmallConvBlock`: a lightweight convolutional block (Conv2D → ReLU).
   - `SmallCNN`: a compact convolutional neural network for CIFAR-10, designed
     for fast training and deployment in FL or edge device scenarios.

2. **Training and Evaluation**
   - `train_model_one_epoch()`: trains a model for a single epoch with progress tracking.
   - `model_one_epoch_losses()`: computes training loss progression across batches.
   - `evaluate_model_on_test()`: evaluates a model on the test dataset and reports accuracy/loss.

3. **Dataset Utilities**
   - `get_cifar10()`: prepares CIFAR-10 train/test DataLoaders with standard normalization and augmentation.
   - `get_homogenous_domains()`: partitions the training set into stratified client datasets
     that maintain approximately equal class distributions (homogeneous domains).
   - `_make_stratified_subsets()`: internal helper for creating balanced per-client splits.

4. **Future Extensions**
   - `_make_heterogenous_subsets()` and `get_heterogenous_domains()` placeholders for
     generating non-IID (heterogeneous) client datasets, e.g., via Dirichlet sampling.

Intended Use:
-------------
This file acts as the shared utility module for the FL framework (used by `fed_model.py`
and `fed_methods.py`).

Example:
--------
    from utils import SmallCNN, get_cifar10, get_homogenous_domains, train_model_one_epoch

    # Prepare data and model
    train_loader, test_loader = get_cifar10(batch_size=64)
    model = SmallCNN()
    
    # Train one epoch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loss, acc = train_model_one_epoch(model, train_loader, criterion, optimizer, "cuda")

    # Create client splits
    client_loaders = get_homogenous_domains(train_loader, clients=5, distributions=[0.2]*5)

Dependencies:
-------------
- torch, torchvision, tqdm, matplotlib
- CIFAR-10 dataset from torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import tqdm
from typing import List, Tuple, Optional
import random
import math
import matplotlib.pyplot as plt

### Basic Model
class SmallConvBlock(nn.Module):
    """
    Small convolutional block: Conv2d -> ReLU.

    Args:
        in_channels (int): input channel count.
        out_channels (int): output channel count.

    Example:
        block = SmallConvBlock(3, 32)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.block[0].in_channels}, out_channels={self.block[0].out_channels})"

class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-10.

    Architecture:
        - SmallConvBlock(3 -> 32)
        - SmallConvBlock(32 -> 64)
        - MaxPool2d(2)
        - AdaptiveAvgPool2d((1,1))
        - Flatten -> Linear(64 -> num_classes)

    Args:
        num_classes (int): number of output classes (default 10).
    """
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

def train_model_one_epoch(model: SmallCNN, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer, device: torch.device):
    """
    Train model for one epoch.

    Returns:
        avg_loss (float): average training loss over the epoch
        accuracy (float): training accuracy (0-1)
    """
    model.train()
    model.to(device)

    for p in model.parameters():
        p.grad = None

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm.tqdm(train_loader, desc="Train", leave=False)
    for batch in pbar:
        inputs, targets = batch
        inputs: torch.Tensor = inputs.to(device, non_blocking=True)
        targets: torch.Tensor = targets.to(device, non_blocking=True)
        
        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, targets)
        loss.backward()
        
        # bookkeeping
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        running_correct += (preds == targets).sum().item()
        running_total += inputs.size(0)

        # running averages for live display
        avg_loss = running_loss / running_total
        avg_acc = running_correct / running_total
        pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{avg_acc:.4f}"})

    if running_total > 0:
        # convert summed grads -> average per-sample grads (match clients)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(running_total)

        all_model = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
        if all_model:
            cvec = torch.cat(all_model)
            print("model avg grad norm:", float(torch.norm(cvec)))

        optimizer.step()
        optimizer.zero_grad()

    avg_loss = running_loss / running_total if running_total > 0 else 0.0
    accuracy = running_correct / running_total if running_total > 0 else 0.0

    return avg_loss, accuracy

def model_one_epoch_losses(model: SmallCNN, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device):
    """
    Train model for one epoch.

    Returns:
        avg_loss (float): average training loss over the epoch
    """
    model.train()
    model.to(device)

    running_loss = 0.0
    running_total = 0

    pbar = tqdm.tqdm(train_loader, desc="Client Training", leave=False)
    for batch in pbar:
        inputs, targets = batch
        inputs: torch.Tensor = inputs.to(device, non_blocking=True)
        targets: torch.Tensor = targets.to(device, non_blocking=True)

        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, targets)
        loss.backward()

        # bookkeeping
        running_loss += loss.item() * inputs.size(0)
        running_total += inputs.size(0)

        # running averages for live display
        avg_loss = running_loss / running_total
        pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

    return running_loss


def evaluate_model_on_test(model: SmallCNN, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device):
    """
    Evaluate model on the test dataset.

    Returns:
        avg_loss (float): average test loss
        accuracy (float): test accuracy (0-1)
    """

    model.eval()
    model.to(device)

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm.tqdm(test_loader, desc="Test ", leave=False)
    with torch.inference_mode():
        for batch in pbar:
            inputs, targets = batch
            inputs: torch.Tensor = inputs.to(device, non_blocking=True)
            targets: torch.Tensor = targets.to(device, non_blocking=True)

            outputs: torch.Tensor = model(inputs)
            loss: torch.Tensor = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            running_correct += (preds == targets).sum().item()
            running_total += inputs.size(0)

            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{avg_acc:.4f}"})

    avg_loss = running_loss / running_total if running_total > 0 else 0.0
    accuracy = running_correct / running_total if running_total > 0 else 0.0

    return avg_loss, accuracy

### Dataset

def get_cifar10(root='./data', train_transform=None, test_transform=None, batch_size=32):
    """
    Create CIFAR-10 DataLoaders.

    Args:
        root (str): directory to download/store CIFAR-10.
        train_transform: torchvision transform applied to training set. If None,
                         a default augmentation + normalization pipeline is used.
        test_transform: torchvision transform applied to test set. If None,
                        a default normalization pipeline is used.
        batch_size (int): batch size for both train and test loaders.

    Returns:
        (train_loader, test_loader): tuple of torch.utils.data.DataLoader
    """
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    train_dataset = CIFAR10(root=root, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root=root, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def _get_targets(dataset: Dataset):
        if hasattr(dataset, "targets"):
            return list(dataset.targets)
        if hasattr(dataset, "labels"):
            return list(dataset.labels)
        raise ValueError("Dataset has no 'targets' or 'labels' attribute.")

def _make_stratified_subsets(dataset, dl, distributions, clients, seed):
    """
    Returns list of index lists (one per client) for `dataset`.
    distributions can be fractions summing to 1 or absolute counts summing to len(dataset).
    """

    if seed is not None:
        random.seed(seed)

    targets = _get_targets(dataset)
    n = len(targets)
    classes = sorted(set(int(t) for t in targets))

    # interpret distributions: fractions -> counts; or absolute counts if sums to n
    s = sum(distributions)
    if abs(s - 1.0) < 1e-8:  # fractions
        counts = [int(round(frac * n)) for frac in distributions]
        # adjust rounding errors to sum to n
        diff = n - sum(counts)
        idx = 0
        while diff != 0:
            counts[idx % clients] += 1 if diff > 0 else -1
            diff = n - sum(counts)
            idx += 1
    elif abs(s - n) < 1e-8 or s == n:  # absolute counts already
        counts = [int(c) for c in distributions]
    else:
        raise ValueError("Incorrect distributions of clients. Must add up to 1.")

    # Prepare per-class index lists
    class_to_indices = {c: [] for c in classes}
    for i, t in enumerate(targets):
        class_to_indices[int(t)].append(i)

    # Build client indices by allocating per class proportionally
    client_indices = [[] for _ in range(clients)]
    # For each class, allocate its indices to clients proportional to counts (client share of total)
    for c in classes:
        idxs = class_to_indices[c]
        m = len(idxs)
        if m == 0:
            continue
        # Determine how many samples of this class go to each client:
        # For client j: desired_j = round( counts[j] * (m/n) )  (proportional share)
        desired = []
        for j in range(clients):
            # fraction of dataset client j should receive
            frac_j = counts[j] / n if n > 0 else 0.0
            desired.append(frac_j * m)
        # Turn desired floats into integer allocations while summing to m
        int_alloc = [int(math.floor(x)) for x in desired]
        remainder = m - sum(int_alloc)
        # distribute remainder to clients with largest fractional parts
        fracs = [(desired[j] - int_alloc[j], j) for j in range(clients)]
        fracs.sort(reverse=True)
        k = 0
        while remainder > 0 and k < len(fracs):
            _, jj = fracs[k]
            int_alloc[jj] += 1
            remainder -= 1
            k += 1
        # now int_alloc sums to m
        start = 0
        for j in range(clients):
            take = int_alloc[j]
            if take > 0:
                client_indices[j].extend(idxs[start:start+take])
                start += take

    # If any client has fewer indices than requested overall counts (due to rounding across classes),
    # we will not artificially move samples — stratification is prioritized.
    return client_indices

def _make_heterogenous_subsets(dataset, distributions, clients, seed, alpha):
    pass

def get_homogenous_domains(
        trainset: DataLoader,
        clients: int, 
        distributions: List[int], 
        seed: int = 42, 
        batch_size: int = 32
        ) -> List[DataLoader]:
    """
    Partition trainset/testset into `clients` subsets according to `distributions`,
    producing stratified splits (per-class) so each client receives an approximately
    equal class distribution relative to its share of the data.

    Args:
        trainset (DataLoader): original training DataLoader (used to access dataset and transforms).
        clients (int): number of clients to create.
        distributions (List[float|int]):
            - Fractions summing to 1.0 (e.g. [0.6,0.3,0.1]), or
          Interpreted per-dataset when fractions; interpreted as absolute counts
          only if the sum equals the dataset size (for that dataset).
        seed (int|None): optional random seed for tie-breaking shuffles.
        base_batch_size (int|None): if provided, use this batch size for returned loaders.

    Returns:
        client_train_loaders: list of DataLoaders (length `clients`).
    """
    
    # build train/test client indices
    train_dataset = trainset.dataset

    train_client_indices = _make_stratified_subsets(train_dataset, trainset, distributions, clients, seed)

    # Build DataLoaders for each client, preserving some DataLoader kwargs
    dl_kwargs = {}
    if hasattr(trainset, "num_workers"):
        dl_kwargs["num_workers"] = trainset.num_workers
    if hasattr(trainset, "pin_memory"):
        dl_kwargs["pin_memory"] = trainset.pin_memory

    train_bs = batch_size or getattr(trainset, "batch_size", 32)

    client_train_loaders = []

    for j in range(clients):
        train_sub = Subset(train_dataset, train_client_indices[j])

        client_train_loaders.append(DataLoader(train_sub, batch_size=train_bs, shuffle=True, **dl_kwargs))

    return client_train_loaders

def get_heterogenous_domains(trainset: DataLoader, clients: int, distributions: List[int], alpha: int) -> List[DataLoader]:
    pass