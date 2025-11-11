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
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import tqdm
from typing import List, Tuple, Optional, Dict
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from fedlab.utils.dataset import BasicPartitioner

  
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
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
        - Flatten -> Linear(64 -> num_classes)

    Args:
        num_classes (int): number of output classes (default 10).
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = nn.Sequential(
            SmallConvBlock(3, 32),
            SmallConvBlock(32, 64),
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

## Model Training Functions/Class

def train_model_one_epoch(model: SmallCNN, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer, device: torch.device, verbose=False):
    """
    Train model for one epoch.

    Returns:
        avg_loss (float): average training loss over the epoch
        accuracy (float): training accuracy (0-1)
    """
    model.train()
    model.to(device)

    optimizer.zero_grad()

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
        
        batch_size = inputs.size(0)
        scaled_loss = loss * batch_size
        scaled_loss.backward()
        
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
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(running_total)
        
        # Debug
        if verbose:
            all_grads = [p.grad.detach().flatten() for p in model.parameters() 
                        if p.grad is not None]
            if all_grads:
                grad_vec = torch.cat(all_grads)
                print(f"Central avg grad norm: {float(torch.norm(grad_vec)):.4f}")

    # Update once
    optimizer.step()

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

def get_cifar10(root='./data', train_transform=None, test_transform=None, batch_size=32, pin_memory=False, num_workers=0):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

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

def _make_heterogenous_subsets(dataset, num_clients, min_require_size, alpha, seed):
    client_indices = BasicPartitioner(dataset.targets, num_clients, partition="noniid-labeldir", dir_alpha=alpha, seed=seed, min_require_size=min_require_size)
    return client_indices

def get_heterogenous_domains(
        trainset: DataLoader,
        clients: int, 
        min_require_size: int, 
        seed: int = 42, 
        batch_size: int = 32,
        alpha: float = 0.1,
        ) -> List[DataLoader]:
    
    train_dataset = trainset.dataset
    train_client_indices = _make_heterogenous_subsets(train_dataset, clients, min_require_size, alpha, seed)

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


### Computing Model Differences

def compute_model_difference(model1: nn.Module, model2: nn.Module, 
                            norm_type: str = 'l2') -> float:
    """
    Compute the norm of the difference between two models' parameters.
    
    Args:
        model1: First model
        model2: Second model
        norm_type: Type of norm to compute ('l2', 'l1', 'linf')
    
    Returns:
        float: Norm of the parameter difference
    
    Example:
        >>> diff = compute_model_difference(fed_model, central_model)
        >>> print(f"Parameter difference: {diff:.6e}")
    """
    diff_list = []
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.shape != p2.shape:
            raise ValueError(f"Model architectures don't match: {p1.shape} vs {p2.shape}")
        
        diff = (p1 - p2).detach().flatten()
        diff_list.append(diff)
    
    # Concatenate all parameter differences
    all_diffs = torch.cat(diff_list)
    
    if norm_type == 'l2':
        return float(torch.norm(all_diffs, p=2))
    elif norm_type == 'l1':
        return float(torch.norm(all_diffs, p=1))
    elif norm_type == 'linf':
        return float(torch.max(torch.abs(all_diffs)))
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
    
def compare_model_parameters(model1: nn.Module, model2: nn.Module, 
                            model1_name: str = "Model 1",
                            model2_name: str = "Model 2",
                            show_top_k: int = 5, verbose=False) -> Dict[str, float]:
    """
    Detailed per-layer comparison of two models.
    
    Args:
        model1: First model (e.g., FedSGD)
        model2: Second model (e.g., Centralized)
        model1_name: Display name for model1
        model2_name: Display name for model2
        show_top_k: Number of layers with largest differences to display
    
    Returns:
        dict: Statistics about the differences
    
    Example:
        >>> stats = compare_model_parameters(fed_model, central_model,
        ...                                   "FedSGD", "Centralized")
    """
    layer_diffs = []
    
    if verbose:
        print(f"\n{'='*70}")
    print(f"Comparing {model1_name} vs {model2_name}")
    if verbose:
        print(f"{'='*70}")
    
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), 
                                         model2.named_parameters()):
        if name1 != name2:
            print(f"Warning: Parameter name mismatch: {name1} vs {name2}")
        
        diff = (p1 - p2).detach()
        
        l2_diff = float(torch.norm(diff, p=2))
        linf_diff = float(torch.max(torch.abs(diff)))
        mean_abs_diff = float(torch.mean(torch.abs(diff)))
        relative_diff = l2_diff / (float(torch.norm(p1, p=2)) + 1e-10)
        
        layer_diffs.append({
            'name': name1,
            'l2': l2_diff,
            'linf': linf_diff,
            'mean_abs': mean_abs_diff,
            'relative': relative_diff,
            'shape': tuple(p1.shape)
        })
    
    # Sort by L2 difference
    layer_diffs_sorted = sorted(layer_diffs, key=lambda x: x['l2'], reverse=True)
    
    # Print top-k largest differences
    print(f"\nTop {show_top_k} layers with largest L2 differences:")
    if verbose:
        print(f"{'-'*70}")
    print(f"{'Layer':<40} {'L2 Diff':>12} {'Rel Diff':>12}")
    if verbose:
        print(f"{'-'*70}")
    
    for i, layer_info in enumerate(layer_diffs_sorted[:show_top_k]):
        print(f"{layer_info['name']:<40} {layer_info['l2']:>12.6e} {layer_info['relative']:>12.6e}")
    
    # Compute overall statistics
    total_l2 = sum(ld['l2'] ** 2 for ld in layer_diffs) ** 0.5
    total_linf = max(ld['linf'] for ld in layer_diffs)
    avg_relative = np.mean([ld['relative'] for ld in layer_diffs])
    
    stats = {
        'total_l2': total_l2,
        'total_linf': total_linf,
        'avg_relative_diff': avg_relative,
        'num_parameters': sum(np.prod(ld['shape']) for ld in layer_diffs)
    }

    if verbose:  
       print(f"\n{'='*70}")
    print(f"Overall Statistics:")
    if verbose:
        print(f"{'='*70}")
    print(f"Total L2 difference:        {stats['total_l2']:.6e}")
    if verbose:
        print(f"Total L-inf difference:     {stats['total_linf']:.6e}")
        print(f"Avg relative difference:    {stats['avg_relative_diff']:.6e}")
    print(f"Total parameters:           {stats['num_parameters']:,}")
    if verbose:
        print(f"{'='*70}\n")
    
    return stats