"""
fed_methods.py
====================
This module defines the base class `FedMethod` and implements the `FedSGD` algorithm 
for Federated Learning (FL) training.

Federated methods define how client updates are aggregated and how the server model 
is updated across communication rounds. This modular design allows easy extension 
to other FL methods such as FedAvg, FedProx, FedSAM, etc.

Classes:
---------
- FedMethod: 
    Abstract base class specifying the FL interface for client/server training, 
    aggregation, and evaluation.

- FedSGD:
    Implements the classic Federated Stochastic Gradient Descent algorithm where:
    - Each client trains on its local data for one epoch using SGD.
    - The server aggregates the model parameters via weighted averaging.
    - Evaluation compares federated and centralized training progress.

Key Functions:
--------------
- exec_client_round(): 
    Syncs clients with the server and trains locally.
- exec_server_round(): 
    Aggregates client models into the server using weighted averaging.
- evaluate_round(): 
    Compares test accuracy/loss between federated and centralized models.
- evaluate_server(): 
    Final evaluation at the end of training.

Usage Example:
--------------
    fed_method = FedSGD(client_weights=[1, 1, 2])
    fed_method.exec_client_round(server, clients, client_dataloaders, device="cuda", lr=0.01)
    fed_method.exec_server_round(clients, server)
    fed_method.evaluate_round(server, central_model, test_loader=test_loader, device="cuda")
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Sequence, Dict
from fed_lib.utils import (
    SmallCNN,
    train_model_one_epoch,
    evaluate_model_on_test,
)

class FedMethod:
    """
    Abstract Class for Fed Methods
    """
    def __init__(self):
        pass

    def exec_server_round(self, clients: List[SmallCNN], server: SmallCNN, **kwargs):
        raise NotImplementedError
    
    def exec_client_round(self, server: SmallCNN, clients: List[SmallCNN], **kwargs):
        raise NotImplementedError

    def evaluate_round(self, server: SmallCNN, central: SmallCNN, **kwargs):
        raise NotImplementedError
    
    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        raise NotImplementedError

class FedSGD(FedMethod):
    """
    Federated SGD aggregator.

    Args:
        client_weights (Optional[Sequence[float]]): optional per-client weights, If None,
            weights default to client dataset sizes (N_i / sum_j N_j).
    """
    def __init__(self, client_weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.client_weights = None if client_weights is None else list(client_weights)

    def _normalize_weights(self, n_clients: int, client_sizes: Optional[List[int]] = None) -> List[float]:
        if client_sizes is not None:
            total_size = sum(client_sizes)
            if total_size > 0:
                return [size / total_size for size in client_sizes]

        if self.client_weights is not None:
            if len(self.client_weights) != n_clients:
                raise ValueError("Length of client_weights must equal number of clients.")
            w = [float(x) for x in self.client_weights]
            s = sum(w)
            if s <= 0.0:
                raise ValueError("Sum of client_weights must be positive.")
            return [x / s for x in w]

        # Fallback: equal weights
        return [1.0 / n_clients] * n_clients

    def exec_server_round(self, clients: List[SmallCNN], server: SmallCNN, **kwargs):
        """
        Execute one federated SGD round: average client gradients (weighted) and update server params.

        Args:
            clients: list of SmallCNNs trained for one epoch.
            server: the server model (nn.Module) whose parameters will be updated.
        """

        n_clients = len(clients)
        device = kwargs['device']
        lr = kwargs.get('lr', 0.001)
        client_sizes = kwargs.get('client_sizes', None)
        verbose = kwargs['verbose']

        if n_clients == 0:
            raise ValueError("clients must contain at least one model.")
        
        # normalize weights
        weights = self._normalize_weights(n_clients)

        server_optimizer: torch.optim.SGD = kwargs.get('server_optimizer', torch.optim.SGD(server.parameters(), lr=lr))

        print("Applying FedSGD on Server")
        if verbose:
            print(f"Aggregating {n_clients} clients with weights: {[f'{w:.3f}' for w in weights]}")

        for p in server.parameters():
            p.grad = None

        with torch.no_grad():
            for p_idx, server_param in enumerate(server.parameters()):
                agg_grad = None

                for client, weight in zip(clients, weights):
                    client_param = list(client.parameters())[p_idx]
                    client_grad = client_param.grad
                    
                    if client_grad is None:
                        continue
                    
                    # Ensure same device
                    if client_grad.device != server_param.device:
                        client_grad = client_grad.to(server_param.device)
                    
                    # Weight the gradient
                    weighted_grad = client_grad * weight
                    
                    if agg_grad is None:
                        agg_grad = weighted_grad.clone()
                    else:
                        agg_grad += weighted_grad

                if agg_grad is not None:
                    server_param.grad = agg_grad

        # Debug: print aggregated gradient norm
        if verbose:
            all_server_grads = [p.grad.detach().flatten() for p in server.parameters() 
                            if p.grad is not None]
            if all_server_grads:
                grad_vec = torch.cat(all_server_grads)
                print(f"Aggregated server grad_norm: {float(torch.norm(grad_vec)):.4f}")

        server_optimizer.step()
        server_optimizer.zero_grad()

    def _train_client(self, client: SmallCNN, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, device):
        client.to(device)
        client.train()

        for p in client.parameters():
            p.grad = None
        
        total_samples = 0.0
        total_loss= 0.0

        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            out = client(inputs)
            loss = criterion(out, targets)
            
            batch_size = inputs.size(0)
            scaled_loss = loss * batch_size
            scaled_loss.backward()

            total_loss += loss.item() * batch_size
            total_samples += inputs.size(0)
        
        if total_samples == 0:
            return 0, 0.0
        
        with torch.no_grad():
            for p in client.parameters():
                if p.grad is not None:
                    p.grad.div_(total_samples)
        
        avg_loss = total_loss / total_samples

        return total_samples, avg_loss

    def exec_client_round(self, server: SmallCNN, clients: List[SmallCNN], client_dataloaders: List[DataLoader], **kwargs):

        device = kwargs['device']
        lr = kwargs['lr']
        verbose = kwargs['verbose']

        criterion = nn.CrossEntropyLoss()

        client_sizes = []

        for i, (client, loader) in enumerate(zip(clients, client_dataloaders)):
            print(f"Training Client {i+1}/{len(clients)}")

            client.load_state_dict(server.state_dict())
            n_samples, avg_loss = self._train_client(client, loader, criterion, device)

            client_sizes.append(n_samples)

            # Debug: print gradient norm
            if verbose:
                all_grads = [p.grad.detach().flatten() for p in client.parameters() 
                            if p.grad is not None]
                if all_grads:
                    grad_vec = torch.cat(all_grads)
                    grad_norm = float(torch.norm(grad_vec))
                    print(f"  Client {i+1}: samples={n_samples}, loss={avg_loss:.4f}, "
                            f"grad_norm={grad_norm:.4f}")
                else:
                    print(f"  Client {i+1}: No gradients computed!")

        kwargs['client_sizes'] = client_sizes


    def evaluate_round(self, server: SmallCNN, central: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss()
        device = kwargs['device']
        test_loader = kwargs['test_loader']

        server_loss, server_acc = evaluate_model_on_test(server, test_loader, criterion, device)
        central_loss, central_acc = evaluate_model_on_test(central, test_loader, criterion, device)

        print(f"FedSGD  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")
        print(f"Central | Test Loss: {central_loss:.4f}, Test Acc: {central_acc*100:.2f}%")

    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        self.evaluate_round(server, central, **kwargs)