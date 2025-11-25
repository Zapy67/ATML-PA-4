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
from torch.utils.data import DataLoader, Subset
from typing import List, Optional, Sequence, Dict
from fed_lib.utils import (
    SmallCNN,
    train_model_one_epoch,
    evaluate_model_on_test,
    compute_model_difference,
    calculate_client_drift_metrics,
)
import random
import numpy as np
import copy


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

        self.round_metrics = {
            'fed_test_acc': [],
            'fed_test_loss': [],
            'central_test_acc': [],
            'central_test_loss': [],
            'client_drift': [],
            'param_difference': []
        }

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

        return [1.0 / n_clients] * n_clients

    def exec_server_round(self, clients: List[SmallCNN], server: SmallCNN, **kwargs):
       
        n_clients = len(clients)
        verbose = kwargs['verbose']
        client_sizes = kwargs.get('client_sizes')

        if n_clients == 0:
            raise ValueError("clients must contain at least one model.")
        
        weights = self._normalize_weights(n_clients, client_sizes)
        print("Applying FedSGD on Server")
        if verbose:
            print(f"Aggregating {n_clients} clients with weights: {[f'{w:.3f}' for w in weights]}")

        server_sd = copy.deepcopy(server.state_dict())
        agg_state_dict = {}

        for k, v in server_sd.items():
            if torch.is_floating_point(v):
                agg_state_dict[k] = v.clone().zero_()
            else:
                agg_state_dict[k] = v.clone()

        with torch.no_grad():
            for client, weight in zip(clients, weights):
                client_sd = client.state_dict()
                for k in agg_state_dict.keys():
                    if not torch.is_floating_point(agg_state_dict[k]):
                        continue                   
                    src = client_sd[k]
                    if src.device != agg_state_dict[k].device:
                        src = src.to(agg_state_dict[k].device)
                    
                    if src.dtype != agg_state_dict[k].dtype:
                        src = src.to(dtype=agg_state_dict[k].dtype)

                    agg_state_dict[k] += src * float(weight)
    
        server.load_state_dict(agg_state_dict)


        # # server_optimizer.zero_grad(set_to_none=True)
        # agg_state_dict = copy.deepcopy(server.state_dict())
        # for key in agg_state_dict.keys():
        #     agg_state_dict[key].zero_()

        # with torch.no_grad():
        #     for client, weight in zip(clients, weights):
        #         client_state_dict = copy.deepcopy(client.state_dict())
        #         for key in agg_state_dict.keys():
        #             # Ensure same device
        #             if client_state_dict[key].device != agg_state_dict[key].device:
        #                 client_state_dict[key] = client_state_dict[key].to(agg_state_dict[key].device)
                        
        #             agg_state_dict[key] += client_state_dict[key] * weight
                
        # server.load_state_dict(agg_state_dict)
        
    def _train_client(self, client: SmallCNN, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, device):
        client.to(device)
        client.train()

        optimizer.zero_grad(set_to_none=True)

        total_loss= 0.0
        total_samples = sum(inputs.size(0) for inputs, _ in dataloader)

        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            out = client(inputs)
            loss = criterion(out, targets) * inputs.size(0) / total_samples
            
            loss.backward()

            total_loss += loss.item() * total_samples
            if (batch_idx+1)==len(dataloader): 
                    print("Optimized")
                    optimizer.step()
                    optimizer.zero_grad()
        
        avg_loss = total_loss / total_samples
        return total_samples, avg_loss

    def exec_client_round(self, server: SmallCNN, clients: List[SmallCNN], client_dataloaders: List[DataLoader], **kwargs):
        device = kwargs['device']
        verbose = kwargs['verbose']
        lr = kwargs.get('lr', 1e-3)
        criterion = nn.CrossEntropyLoss()
        client_sizes = []

        for i, (client, loader) in enumerate(zip(clients, client_dataloaders)):
            print(f"Training Client {i+1}/{len(clients)}")
            params = copy.deepcopy(server.state_dict())
            client.load_state_dict(params)
            optimizer = torch.optim.SGD(client.parameters(), lr=lr)
            n_samples, avg_loss = self._train_client(client, loader, criterion, optimizer, device)

            client_sizes.append(n_samples)

            # Debug: print gradient norm
            # if verbose:
            #     all_grads = [p.grad.detach().flatten() for p in client.parameters() 
            #                 if p.grad is not None]
            #     if all_grads:
            #         grad_vec = torch.cat(all_grads)
            #         grad_norm = float(torch.norm(grad_vec))
            #         print(f"  Client {i+1}: samples={n_samples}, loss={avg_loss:.4f}, "
            #                 f"grad_norm={grad_norm:.4f}")
            #     else:
            #         print(f"  Client {i+1}: No gradients computed!")

        kwargs['client_sizes'] = client_sizes

    def evaluate_round(self, server: SmallCNN, central: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss()
        device = kwargs['device']
        test_loader = kwargs['test_loader']

        server_loss, server_acc = evaluate_model_on_test(server, test_loader, criterion, device)
        central_loss, central_acc = evaluate_model_on_test(central, test_loader, criterion, device)

        self.round_metrics['fed_test_acc'].append(server_acc)
        self.round_metrics['fed_test_loss'].append(server_loss)
        self.round_metrics['central_test_acc'].append(central_acc)
        self.round_metrics['central_test_loss'].append(central_loss)
        param_diff = compute_model_difference(server, central, norm_type='l2')
        self.round_metrics['param_difference'].append(param_diff)

        print(f"FedSGD  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")
        print(f"Central | Test Loss: {central_loss:.4f}, Test Acc: {central_acc*100:.2f}%")

    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        self.evaluate_round(server, central, **kwargs)


class FedAvg(FedMethod):
    def __init__(self, local_epochs: int = 5, aggregation_steps: int=32,
                 client_weights: Optional[Sequence[float]] = None,
                 sample_fraction: float = 1.0):
        super().__init__()
        self.local_epochs = local_epochs
        self.aggregation_steps = aggregation_steps
        self.client_weights = None if client_weights is None else list(client_weights)
        self.sample_fraction = sample_fraction
        self.selected_indices = None
       

        self.round_metrics = {
            'fed_test_acc': [],
            'fed_test_loss': [],
            'client_drift': [],
        }

    def _sample_clients(self, n_clients: int, seed: Optional[int] = None) -> List[int]:
        n_sample = max(1, int(n_clients * self.sample_fraction))
        if seed is not None:
            random.seed(seed)
        return random.sample(range(n_clients), n_sample)
    
    def debug_output(self, model):
        with torch.no_grad():
            params = [param.detach().flatten() for param in model.parameters() 
                                        if param is not None]
            if params:
                param_vector = torch.cat(params)
                print(f"Aggregated server grad_norm: {torch.norm(param_vector).item():.4f}")
    

    def _train_client_local(self, client: SmallCNN, dataloader: DataLoader,
                           criterion: nn.CrossEntropyLoss, lr: float, device: torch.device) -> int:
        client.to(device)
        client.train()
        n_samples = len(dataloader.dataset.indices)
        n_batches = len(dataloader)
        total_loss_accumulated = 0
        step = np.ceil(n_batches/self.aggregation_steps)
        step = max(1, step)

        optimizer = torch.optim.SGD(client.parameters(), lr=lr)
        optimizer.zero_grad(set_to_none=True)
        buffer = list()
            
        def forward_pass():
            total_loss = 0.0
            n_samples = sum(inputs.size(0) for inputs, _ in buffer)
            optimizer.zero_grad(set_to_none=True)
            for inputs, targets in buffer:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = client(inputs) 
                loss = criterion(predictions, targets) * inputs.size(0) / n_samples
                loss.backward()
                total_loss += loss.item() * n_samples

            return total_loss

        for _ in range(self.local_epochs):
            for batch_idx, batch in enumerate(dataloader):  
                buffer.append(batch)
                if len(buffer) == step or (batch_idx+1) == len(dataloader):    
                    loss = forward_pass()
                    optimizer.step()
                    total_loss_accumulated += loss
                    buffer.clear()
            
        return n_samples*self.local_epochs

    def exec_client_round(self, server: SmallCNN, clients: List[SmallCNN],
                         client_dataloaders: List[DataLoader], **kwargs):
        
        device = kwargs['device']
        lr = kwargs['lr']
        verbose = kwargs.get('verbose', False)
        current_round = kwargs.get('current_round', 0) 
        criterion = nn.CrossEntropyLoss()
        n_clients = len(clients)
        selected_indices = self._sample_clients(n_clients, seed=current_round)
        client_sizes = []
        
        for idx in selected_indices:
            client = clients[idx]
            loader = client_dataloaders[idx]
            print(f"Training Client {idx+1}/{n_clients} for {self.aggregation_steps} steps")
            client.load_state_dict(copy.deepcopy(server.state_dict()))
            n_samples = self._train_client_local(client, loader, criterion, lr, device)
            client_sizes.append(n_samples)
            if verbose:
                print(f"  Client {idx+1}: trained on {n_samples} samples")
        
        kwargs['client_sizes'] = client_sizes
        kwargs['selected_indices'] = selected_indices
        self.selected_indices = selected_indices

    def exec_server_round(self, clients: List[SmallCNN], server: SmallCNN, **kwargs):
        selected_indices = kwargs.get('selected_indices', list(range(len(clients))))
        if self.selected_indices is not None:
            selected_indices = self.selected_indices
        n_selected = len(selected_indices)
        total = np.array([self.client_weights[idx] for idx in selected_indices])
        weights = total/sum(total)
        selected_clients = [clients[client_idx] for client_idx in selected_indices]

        drift_summary = calculate_client_drift_metrics(server, selected_clients ,show_top_k=n_selected, verbose=True)
        self.round_metrics['client_drift'].append(drift_summary['mean_client_drift'])
       
        server_sd = copy.deepcopy(server.state_dict())
        agg_state = {}
        for k, v in server_sd.items():
            if torch.is_floating_point(v):
                agg_state[k] = v.clone().zero_()
            else:
                agg_state[k] = v.clone()

        
        with torch.no_grad():
            for client, weight in zip(selected_clients, weights):
                client_sd = client.state_dict()
                
                for k in agg_state.keys():    
                    src = client_sd[k]
                    if src.device != agg_state[k].device:
                        src = src.to(agg_state[k].device)
                    
                    if not torch.is_floating_point(agg_state[k]):
                        continue
                
                    agg_state[k] += src * float(weight)

        server.load_state_dict(agg_state)
        self.selected_indices = None
        
        

    def evaluate_round(self, server: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss()
        device = kwargs['device']
        test_loader = kwargs['test_loader']
        server_loss, server_acc = evaluate_model_on_test(server, test_loader, criterion, device)
        self.round_metrics['fed_test_acc'].append(server_acc)
        self.round_metrics['fed_test_loss'].append(server_loss)
        print(f"FedAvg  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")

    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        self.evaluate_round(server, **kwargs)

    def get_metrics(self):
        return self.round_metrics


class FedSAM(FedAvg):
    def __init__(self, local_epochs: int = 5, aggregation_steps: int=32,
                 client_weights: Optional[Sequence[float]] = None,
                 sample_fraction: float = 1.0, rho: float=1e-3):
        
        super().__init__(local_epochs, aggregation_steps,
                 client_weights ,
                 sample_fraction)
        
        self.rho = rho
    
    def _train_client_local(self, client: SmallCNN, dataloader: DataLoader,
                           criterion: nn.CrossEntropyLoss, lr: float, device: torch.device) -> int:
        client.to(device)
        client.train()
       
        n_batches = len(dataloader)
        step = np.ceil(n_batches/self.aggregation_steps)
        step = max(1, step)
        optimizer = torch.optim.SGD(client.parameters(), lr=lr)
        optimizer.zero_grad(set_to_none=True)
        total_loss_accumulated = 0.0
        buffer = list()

        def forward_pass():
            total_loss = 0.0
            n_samples = sum(inputs.size(0) for inputs, _ in buffer)
            optimizer.zero_grad(set_to_none=True)
            for inputs, targets in buffer:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = client(inputs) 
                loss = criterion(predictions, targets) * inputs.size(0) / n_samples
                loss.backward()
                total_loss += loss.item() * n_samples

            return total_loss


        for batch_idx, batch in enumerate(dataloader):  
            buffer.append(batch)
            if len(buffer) == step or (batch_idx+1) == len(dataloader): 

                original_params = [param.clone() for param in client.parameters()]
                forward_pass()
                norm = 0.0
                for model_param in client.parameters():
                    if model_param.grad is not None:
                        grad = model_param.grad
                        norm += torch.sum(grad * grad)
            
                scale = self.rho / (torch.sqrt(norm) + 1e-12)

                with torch.no_grad():
                    for model_param in client.parameters():
                        if model_param.grad is not None:
                            model_param.add_(model_param.grad * scale)
                
                loss = forward_pass()
       
                with torch.no_grad():
                    for model_param, original_p in zip(client.parameters(), original_params):
                        model_param.copy_(original_p)
            
                optimizer.step()
                total_loss_accumulated += loss
                buffer.clear()
        
        total_samples_processed = len(dataloader.dataset.indices)
        average_loss = total_loss_accumulated / total_samples_processed
        return total_samples_processed, average_loss
    
    def evaluate_round(self, server: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss()
        device = kwargs['device']
        test_loader = kwargs['test_loader']
        server_loss, server_acc = evaluate_model_on_test(server, test_loader, criterion, device)
        self.round_metrics['fed_test_acc'].append(server_acc)
        self.round_metrics['fed_test_loss'].append(server_loss)
        print(f"FedSAM  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")

class FedGH(FedAvg):
    def __init__(self, 
             local_epochs: int = 5, aggregation_steps: int=32,
             client_weights: Optional[Sequence[float]] = None,
             sample_fraction: float = 1.0):
    
        super().__init__(local_epochs, aggregation_steps, client_weights, sample_fraction)

    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        params = [p.data.view(-1) for p in model.parameters() if p.requires_grad]
        return torch.cat(params)

    def _set_flat_params(self, model: nn.Module, flat_params: torch.Tensor):
        offset = 0
        for p in model.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
                offset += numel

    def harmonize_gradients(self, grads_list: List[torch.Tensor]) -> List[torch.Tensor]:
        num_clients = len(grads_list)
        harmonized_list = [g.clone().detach() for g in grads_list]
      
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                g_i = grads_list[i]
                g_j = grads_list[j]
                
                dot = torch.dot(g_i, g_j)
                
              
                if dot < 0:
                    norm_i_sq = torch.dot(g_i, g_i)
                    norm_j_sq = torch.dot(g_j, g_j)
                    
                    if norm_j_sq > 1e-6 :
                        proj_i_on_j = (dot / (norm_j_sq + 1e-6 )) * g_j
                        harmonized_list[i] -= proj_i_on_j                     

                    if norm_i_sq > 1e-6 :
                        proj_j_on_i = (dot / (norm_i_sq + 1e-6 )) * g_i
                        harmonized_list[j] -= proj_j_on_i 

        return harmonized_list

    def exec_server_round(self,
                          clients: List[nn.Module],
                          server: nn.Module,
                          **kwargs):
        
    
        selected_indices = kwargs.get('selected_indices', list(range(len(clients))))
        n_selected = len(selected_indices)
        selected_clients = [clients[client_idx] for client_idx in selected_indices]
        drift_summary = calculate_client_drift_metrics(server, selected_clients ,show_top_k=n_selected, verbose=True)
        self.round_metrics['client_drift'].append(drift_summary['mean_client_drift'])

        server_flat = self._get_flat_params(server)
        pseudo_gradients = []
        
        for client_idx in selected_indices:
            client_flat = self._get_flat_params(clients[client_idx])
            pseudo_grad = server_flat - client_flat
            pseudo_gradients.append(pseudo_grad)

        harmonized_grads = self.harmonize_gradients(pseudo_gradients)
        total = np.array([self.client_weights[idx] for idx in selected_indices])
        aggregation_weights = total/sum(total)
        
        accumulated_grad = torch.zeros_like(server_flat)
        
        for weight, h_grad in zip(aggregation_weights, harmonized_grads):
            accumulated_grad += weight * h_grad

        new_server_params = server_flat - accumulated_grad
        self._set_flat_params(server, new_server_params)

    def evaluate_round(self, server: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss()
        device = kwargs['device']
        test_loader = kwargs['test_loader']
        server_loss, server_acc = evaluate_model_on_test(server, test_loader, criterion, device)
        self.round_metrics['fed_test_acc'].append(server_acc)
        self.round_metrics['fed_test_loss'].append(server_loss)
        print(f"FedGH  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")

class FedProx(FedAvg):
    """
        Implementation of the FedProx algorithm (Li et al., MLSys 2020).

        FedProx extends FedAvg by adding a proximal term to each client's local
        objective, penalizing deviation from the global model parameters. This
        stabilizes training and improves convergence under heterogeneous (non-IID)
        client data distributions. When μ = 0, FedProx reduces to FedAvg.
    """

    def __init__(self, local_epochs: int = 5, aggregation_steps: int=32,
                 client_weights: Optional[Sequence[float]] = None,
                 sample_fraction: float = 1.0, mu: float = 0.5):
        
        super().__init__(local_epochs, aggregation_steps,
                 client_weights ,
                 sample_fraction)
        
        self.mu = mu

    def _train_client_local(self, client: SmallCNN, dataloader: DataLoader, 
                           criterion: nn.CrossEntropyLoss, lr: float, device: torch.device) -> int:
        client.to(device)
        client.train()

        n_batches = len(dataloader)
        step = np.ceil(n_batches/self.aggregation_steps)
        step = max(1, step)
        optimizer = torch.optim.SGD(client.parameters(), lr=lr)
        optimizer.zero_grad(set_to_none=True)
        total_loss_accumulated = 0.0
        buffer = list()

        theta_global = [p.detach().clone().to(device) for p in client.parameters()]

        def forward_pass():
            total_loss = 0.0
            loss = torch.zeros(1, device=device)
            n_samples = sum(inputs.size(0) for inputs, _ in buffer)
            optimizer.zero_grad(set_to_none=True)
            for inputs, targets in buffer:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = client(inputs) 
                # loss: torch.Tensor = criterion(predictions, targets) * inputs.size(0) / n_samples
                loss += criterion(predictions, targets) * inputs.size(0) / n_samples
                # loss.backward()
                # total_loss += loss.item() * n_samples

            # return total_loss
            return loss
        
        for batch_idx, batch in enumerate(dataloader):  
            buffer.append(batch)
            if len(buffer) == step or (batch_idx+1) == len(dataloader): 

                loss: torch.Tensor = forward_pass()

                # WIP
                proximal_term = torch.zeros(1, device=device)
                for p, p_g in zip(client.parameters(), theta_global):
                    proximal_term += self.mu / 2 * (p - p_g).pow(2).sum()

                loss += proximal_term
                loss.backward()
                optimizer.step()
                # total_loss_accumulated += loss + proximal_term.item()
                total_loss_accumulated += loss
                buffer.clear()
        
        total_samples_processed = len(dataloader.dataset.indices)
        average_loss = total_loss_accumulated / total_samples_processed
        return total_samples_processed, average_loss
    
    def evaluate_round(self, server: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss()
        device = kwargs['device']
        test_loader = kwargs['test_loader']
        server_loss, server_acc = evaluate_model_on_test(server, test_loader, criterion, device)
        self.round_metrics['fed_test_acc'].append(server_acc)
        self.round_metrics['fed_test_loss'].append(server_loss)
        print(f"FedProx  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")