"""
fed_model.py
====================
This module defines the core `Federation` class that simulates Federated Learning (FL)
training using client-server coordination.

Each client and the central server share the same CNN architecture (`SmallCNN`), 
and model updates are managed through a chosen federated method (`FedMethod` subclass), 
such as FedSGD, FedAvg, or custom strategies.

Key functionalities:
- Initializes multiple clients and a server model
- Distributes training data across clients (via `get_homogenous_domains`)
- Performs federated training rounds with client and server updates
- Compares federated training performance with centralized training
*To-Do:
- Robust Evaluation of Federation, such as client drift, domain shift, etc.

Dependencies:
- torch, torch.nn
- utils.py for dataset handling and training utilities
- fed_methods.py for specific FL algorithms

Usage:
    federation = Federation(num_clients=5, federate_method=FedAvg(), domains=[0,1,2,3,4], device="cuda")
    federation.train(rounds=10, lr=0.01)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from utils import (
    SmallCNN,
    train_model_one_epoch,
    evaluate_model_on_test,
    get_homogenous_domains,
    get_cifar10,
)
from fed_methods import (
    FedMethod,
)
from typing import List

from copy import deepcopy

class Federation:
    def __init__(self, num_clients: int, federate_method: FedMethod, domains: List[int], device):
        
        self.clients = [SmallCNN().to(device) for _ in range(num_clients)]
        self.server = SmallCNN().to(device)
        self.device = device

        trainloader, self.test_loader = get_cifar10()

        self.client_dataloaders = get_homogenous_domains(trainloader, num_clients, domains)

        # Centralized dataset for comparison
        self.centralized_train_loader = deepcopy(trainloader)

        self.federated_method = federate_method

    def train(self, rounds: int, lr = 0.001, **kwargs):
        criterion = nn.CrossEntropyLoss()
        
        central_model = deepcopy(self.server)
        central_optimizer = torch.optim.SGD(central_model.parameters(), lr=lr)
        server_optimizer = torch.optim.SGD(self.server.parameters(), lr=lr)

        kwargs['device'] = self.device
        kwargs['lr'] = lr
        kwargs['test_loader'] = self.test_loader
        kwargs['server_optimizer'] = server_optimizer

        for round in range(rounds):
            print(f"\n--- Round {round+1}/{rounds} ---")
            # Train

            self.federated_method.exec_client_round(self.server, self.clients, self.client_dataloaders, **kwargs)
            
            self.federated_method.exec_server_round(self.clients, self.server, **kwargs)

            train_model_one_epoch(central_model, self.centralized_train_loader, criterion, central_optimizer, self.device)

            # Test
            self.federated_method.evaluate_round(self.server, central_model, **kwargs)
            
        
        print("Training Complete!")

        self.federated_method.evaluate_server(self.server, central_model, **kwargs)

    def test(self):
        raise NotImplementedError