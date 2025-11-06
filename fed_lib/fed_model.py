import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    def __init__(self, clients: int, federate_method: FedMethod, domains: List[int]):
        
        self.clients = [SmallCNN() for _ in clients]
        self.server = SmallCNN()

        trainset, testset = get_cifar10()

        self.domains = get_homogenous_domains(trainset, clients, domains)
        self.fed = federate_method

    def train(self, rounds: int, parallel_centralized=True, lr = 0.001, device = None):
        criterion = nn.CrossEntropyLoss()
        optimizers = [torch.optim.SGD(self.server.parameters(), lr=lr) for _ in self.clients]
        central_model = None
        central_optimizer = None
        central_dataset = None
        test_loader = None

        if parallel_centralized == True:
            central_model = SmallCNN()
            central_dataset, test_loader = get_cifar10()
            central_optimizer = torch.optim.SGD(central_model.parameters(), lr=lr)
        else:
            _, test_loader = get_cifar10()

        for round in range(rounds):
            # Train
            for i, client in enumerate(self.clients):
                client = deepcopy(self.server)
                train_model_one_epoch(client, self.domains[i], criterion, optimizers[i], device=device)
            
            self.fed.exec_round(self.clients, self.server)

            if central_model is not None:
                train_model_one_epoch(central_model, central_dataset, criterion, central_optimizer, device=device)

            # Test
            server_loss, server_acc = evaluate_model_on_test(self.server, test_loader, criterion, device)
            if central_model is not None:
                central_loss, central_acc = evaluate_model_on_test(central_model, test_loader, criterion, device)

            print(f"Server Loss ({server_loss:.2f}) and Accuracy ({server_acc*100:.2f}%) on {round}'th round.")
            if central_model is not None:
                print(f"Central Model Loss ({server_loss:.2f}) and Accuracy ({server_acc*100:.2f}%) on {round}'th round.")

        server_loss, server_acc = evaluate_model_on_test(self.server, test_loader, criterion, device)

        if central_model is not None:
            central_loss, central_acc = evaluate_model_on_test(central_model, test_loader, criterion, device)

        print(f"Server Loss ({server_loss:.2f}) and Accuracy ({server_acc*100:.2f}%) after {rounds} rounds.")
        if central_model is not None:
            print(f"Central Model Loss ({server_loss:.2f}) and Accuracy ({server_acc*100:.2f}%) after {rounds} rounds.")
