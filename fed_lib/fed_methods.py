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

        # Track metrics
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
            # If total_size is 0, fallback to equal weights

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
        verbose = kwargs['verbose']
        client_sizes = kwargs.get('client_sizes')

        if n_clients == 0:
            raise ValueError("clients must contain at least one model.")
        
        # normalize weights
        weights = self._normalize_weights(n_clients, client_sizes)

        # server_optimizer: torch.optim.SGD = kwargs['server_optimizer']

        print("Applying FedSGD on Server")
        if verbose:
            print(f"Aggregating {n_clients} clients with weights: {[f'{w:.3f}' for w in weights]}")

        # server_optimizer.zero_grad(set_to_none=True)
        agg_state_dict = copy.deepcopy(server.state_dict())
        for key in agg_state_dict.keys():
            agg_state_dict[key].zero_()

        with torch.no_grad():
            for client, weight in zip(clients, weights):
                client_state_dict = copy.deepcopy(client.state_dict())
                for key in agg_state_dict.keys():
                    # Ensure same device
                    if client_state_dict[key].device != agg_state_dict[key].device:
                        client_state_dict[key] = client_state_dict[key].to(agg_state_dict[key].device)
                        
                    agg_state_dict[key] += client_state_dict[key] * weight
                
        server.load_state_dict(agg_state_dict)
        
    def _train_client(self, client: SmallCNN, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, device):
        client.to(device)
        client.train()

        optimizer.zero_grad(set_to_none=True)
        
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

        optimizer.step()
        
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


class FedSAM(FedMethod):
    def __init__(self, 
             sam_rho: float, 
             num_local_steps: int, 
             client_aggregation_weights: list[float]):
    
        super().__init__()
        self.client_weights =  client_aggregation_weights 
        self.rho = sam_rho
        self.K = num_local_steps

        # Track metrics
        self.round_metrics = {
            'fed_test_acc': [],
            'fed_test_loss': [],
            'central_test_acc': [],
            'central_test_loss': [],
            'client_drift': [],
            'param_difference': []
        }
    
    def debug_output(self, model):
        with torch.no_grad():
            params = [param.detach().flatten() for param in model.parameters() 
                                        if param is not None]
            if params:
                param_vector = torch.cat(params)
                print(f"Aggregated server grad_norm: {torch.norm(param_vector).item():.4f}")
    
    def exec_server_round(self, 
                        local_models: List[SmallCNN], 
                        global_model: SmallCNN, 
                        **kwargs):
       

        num_clients = len(local_models)
        verbose = kwargs['verbose']
    
        if num_clients == 0:
            raise ValueError("local_models must contain at least one model.")
        
        aggregation_weights = self.client_weights

        if verbose:
            print(f"Aggregating {num_clients} clients with weights: {[f'{weight:.3f}' for weight in aggregation_weights]}")
        
        calculate_client_drift_metrics(global_model, local_models, show_top_k=5, verbose=True)
     
        with torch.no_grad():
            keys = global_model.state_dict().keys()
            client_weights_list = []

            for agg_weight, local_model in zip(aggregation_weights, local_models):
                local_params = list(local_model.state_dict().values())
                client_weights_list.append([param * agg_weight for param in local_params])

            
            aggregated_params = [sum(param_group) for param_group in zip(*client_weights_list)]

            global_state_dict = global_model.state_dict()
            for key, aggregated_tensor in zip(keys, aggregated_params):
                global_state_dict[key].copy_(aggregated_tensor)
               
        if verbose:
            self.debug_output(global_model)

    def _train_client(self, 
                  local_model: SmallCNN, 
                  local_dataloader: DataLoader,
                  criterion: nn.CrossEntropyLoss,
                  device: torch.device, 
                  **kwargs):
    
        local_model.to(device)
        local_model.train()

        learning_rate = kwargs['lr']
        momentum = kwargs['momentum']
        weight_decay = kwargs['weight_decay']

        total_samples_processed = 0.0
        total_loss_accumulated = 0.0
        num_steps = 0
        
        local_optimizer = torch.optim.SGD(
            local_model.parameters(), 
            lr=learning_rate, 
            momentum=momentum, 
            weight_decay=weight_decay
        )

        for data_batch in local_dataloader:
            if num_steps == self.K:
                break
            num_steps +=1
            
            batch_inputs, batch_targets = data_batch
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            def calculate_gradients_closure():
                local_optimizer.zero_grad(set_to_none=True)
                predictions = local_model(batch_inputs) 
                loss = criterion(predictions, batch_targets)
                loss.backward()
                return loss
             
            original_params = [param.clone() for param in local_model.parameters()]
            
            calculate_gradients_closure()
        
            norm = 0.0
            for model_param in local_model.parameters():
                if model_param.grad is not None:
                    grad = model_param.grad
                    norm += torch.sum(grad * grad)
            
            scale = self.rho / (torch.sqrt(norm) + 1e-12)

            with torch.no_grad():
                for model_param in local_model.parameters():
                    if model_param.grad is not None:
                        model_param.add_(model_param.grad * scale)
            
            loss = calculate_gradients_closure()
       
            with torch.no_grad():
                for model_param, original_p in zip(local_model.parameters(), original_params):
                    model_param.copy_(original_p)
            
            local_optimizer.step()
            total_loss_accumulated += loss.item()
            total_samples_processed += batch_inputs.size(0)
        
        if total_samples_processed == 0:
            return 0, 0.0
         
        average_loss = total_loss_accumulated / total_samples_processed

        return total_samples_processed, average_loss

    def exec_client_round(self, server: SmallCNN, clients: List[SmallCNN], client_dataloaders: List[DataLoader], **kwargs):

        device = kwargs['device']
        verbose = kwargs['verbose']
        lr = kwargs['lr']
        momentum = kwargs['momentum']
        weight_decay = kwargs['weight_decay']

        criterion = nn.CrossEntropyLoss(reduction='sum')

        client_sizes = []
        client_losses = []

        for i, (client, loader) in enumerate(zip(clients, client_dataloaders)):
            print(f"Training Client {i+1}/{len(clients)}")

            params = copy.deepcopy(server.state_dict())
            client.load_state_dict(params)
            n_samples, avg_loss = self._train_client(client, loader, criterion, device, lr=lr, momentum=momentum, weight_decay=weight_decay)

            client_sizes.append(n_samples)
            client_losses.append(avg_loss)

            if verbose:
                self.debug_output(client)

        kwargs['client_losses'] = client_losses
        kwargs['client_sizes'] = client_sizes


    def evaluate_round(self, server: SmallCNN, central: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss(reduction='sum')
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

        print(f"FedSAM  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")
        print(f"Central | Test Loss: {central_loss:.4f}, Test Acc: {central_acc*100:.2f}%")

    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        self.evaluate_round(server, central, **kwargs)

class FedGH(FedMethod):
    def __init__(self, 
                 num_local_steps: int, 
                 client_aggregation_weights: list[float]):
    
        super().__init__()
        self.client_weights = client_aggregation_weights 
        self.K = num_local_steps

        # Track metrics
        self.round_metrics = {
            'fed_test_acc': [],
            'fed_test_loss': [],
            'central_test_acc': [],
            'central_test_loss': [],
            'client_drift': [],
            'param_difference': []
        }

    def debug_output(self, model):
        with torch.no_grad():
            params = [param.detach().flatten() for param in model.parameters() 
                                            if param is not None]
            if params:
                param_vector = torch.cat(params)
                print(f"Aggregated server grad_norm: {torch.norm(param_vector).item():.4f}")
    
    def exec_server_round(self, 
                          local_models: List[nn.Module], 
                          global_model: nn.Module, 
                          **kwargs):
       
        num_clients = len(local_models)
        verbose = kwargs.get('verbose', False) # Safe get
    
        if num_clients == 0:
            raise ValueError("local_models must contain at least one model.")
        
        aggregation_weights = self.client_weights

        if verbose:
            print(f"Aggregating {num_clients} clients with weights: {[f'{weight:.3f}' for weight in aggregation_weights]}")

        def harmonize_gradients(grads_list):           
            harmonized_grads = []
            for i, g_i in enumerate(grads_list):
                for j, g_j in enumerate(harmonized_grads):
                    dot = torch.dot(g_i, g_j)
                    if dot < 0:
                        norm_sq = torch.dot(g_j, g_j)
                        if norm_sq > 0:
                            g_i -= (dot / norm_sq) * g_j
                harmonized_grads.append(g_i)
            return harmonized_grads
       
        with torch.no_grad():
            global_params_flat = torch.cat([p.flatten() for p in global_model.parameters()])
            client_updates = []
            
            for local_model in local_models:
                local_params_flat = torch.cat([p.flatten() for p in local_model.parameters()])
                update = local_params_flat - global_params_flat
                client_updates.append(update)

            harmonized_updates = harmonize_gradients(client_updates)

            weighted_updates = torch.zeros_like(global_params_flat)
            for agg_weight, h_update in zip(aggregation_weights, harmonized_updates):
                weighted_updates += h_update * agg_weight

            new_global_params = global_params_flat + weighted_updates

            start_idx = 0
            global_state_dict = global_model.state_dict()
            for key, param in global_state_dict.items():
                length = param.numel()
                new_tensor = new_global_params[start_idx:start_idx+length].view(param.shape)
                global_state_dict[key].copy_(new_tensor)
                start_idx += length
               
        if verbose:
            self.debug_output(global_model)

    def _train_client(self, 
                  local_model: nn.Module, 
                  local_dataloader: DataLoader,
                  criterion: nn.CrossEntropyLoss,
                  device: torch.device, 
                  **kwargs):
    
        local_model.to(device)
        local_model.train()

        learning_rate = kwargs['lr']
        momentum = kwargs['momentum']
        weight_decay = kwargs['weight_decay']

        total_samples_processed = 0.0
        total_loss_accumulated = 0.0
        num_steps = 0
        counter = 0
        batch_steps = max(1, int(len(local_dataloader)/ self.K))
        leftover = len(local_dataloader) - batch_steps * self.K
        curr_samples = 0.0
        
        local_optimizer = torch.optim.SGD(
            local_model.parameters(), 
            lr=learning_rate, 
            momentum=momentum, 
            weight_decay=weight_decay
        )

        local_optimizer.zero_grad(set_to_none=True)

        for data_batch in local_dataloader:
            counter +=1
            
            batch_inputs, batch_targets = data_batch
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            predictions = local_model(batch_inputs) 
            loss = criterion(predictions, batch_targets)
            loss.backward()
            total_loss_accumulated += loss.item()
            total_samples_processed += batch_inputs.size(0)
            if counter == (batch_steps + (1 if num_steps < leftover else 0)):
                local_optimizer.step()
                local_optimizer.zero_grad(set_to_none=True)
                counter = 0
                num_steps += 1
                if leftover > 0: leftover -= 1

        if num_steps < self.K:
            local_optimizer.step()
            local_optimizer.zero_grad(set_to_none=True)
        
        if total_samples_processed == 0:
            return 0, 0.0
         
        average_loss = total_loss_accumulated / total_samples_processed

        return total_samples_processed, average_loss

    def exec_client_round(self, server: nn.Module, clients: List[nn.Module], client_dataloaders: List[DataLoader], **kwargs):

        device = kwargs['device']
        verbose = kwargs['verbose']
        lr = kwargs['lr']
        momentum = kwargs['momentum']
        weight_decay = kwargs['weight_decay']

        criterion = nn.CrossEntropyLoss(reduction='sum')

        client_sizes = []
        client_losses = []

        for i, (client, loader) in enumerate(zip(clients, client_dataloaders)):

            params = copy.deepcopy(server.state_dict())
            client.load_state_dict(params)
            n_samples, avg_loss = self._train_client(client, loader, criterion, device, lr=lr, momentum=momentum, weight_decay=weight_decay)

            client_sizes.append(n_samples)
            client_losses.append(avg_loss)

            if verbose:
                self.debug_output(client)

        kwargs['client_losses'] = client_losses
        kwargs['client_sizes'] = client_sizes


    def evaluate_round(self, server: nn.Module, central: nn.Module, **kwargs):
        criterion = nn.CrossEntropyLoss(reduction='sum')
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

        print(f"FedGH   | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")
        print(f"Central | Test Loss: {central_loss:.4f}, Test Acc: {central_acc*100:.2f}%")
        
    def evaluate_server(self, server: nn.Module, central: nn.Module, **kwargs):
        self.evaluate_round(server, central, **kwargs)

class FedAvg(FedMethod):
    """
    Federated Averaging (FedAvg) algorithm.

    Key differences from FedSGD:
    - Clients train for K > 1 local epochs before communication
    - Server aggregates model parameters (not gradients)
    - Supports client sampling (partial participation)
    """
    def __init__(self, local_epochs: int = 5,
                 client_weights: Optional[Sequence[float]] = None,
                 sample_fraction: float = 1.0):
        super().__init__()
        self.local_epochs = local_epochs
        self.client_weights = None if client_weights is None else list(client_weights)
        self.sample_fraction = sample_fraction

        # Track metrics
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

    def _sample_clients(self, n_clients: int, seed: Optional[int] = None) -> List[int]:
        n_sample = max(1, int(n_clients * self.sample_fraction))
        if seed is not None:
            random.seed(seed)
        return random.sample(range(n_clients), n_sample)

    def _train_client_local(self, client: SmallCNN, dataloader: DataLoader,
                           criterion: nn.CrossEntropyLoss, lr: float, device: torch.device) -> int:
        client.to(device)
        client.train()
        optimizer = torch.optim.SGD(client.parameters(), lr=lr)
        total_samples = 0

        num_steps = 0
        counter = 0
        batch_steps = max(1, int(len(dataloader)/ self.local_epochs))
        leftover = len(dataloader) - batch_steps * self.local_epochs
        curr_samples = 0.0

        optimizer.zero_grad(set_to_none=True)

        for inputs, targets in dataloader:
            counter +=1
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = client(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            curr_samples += inputs.size(0)
            total_samples += inputs.size(0)

            if counter == (batch_steps + (1 if num_steps < leftover else 0)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                counter = 0
                num_steps += 1
                if leftover > 0: leftover -= 1

        if num_steps < self.local_epochs:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return total_samples

    def _compute_client_drift(self, clients: List[SmallCNN], server: SmallCNN,
                             selected_indices: List[int]) -> float:
        total_drift = 0.0
        with torch.no_grad():
            for idx in selected_indices:
                drift = 0.0
                for p_client, p_server in zip(clients[idx].parameters(), server.parameters()):
                    drift += torch.norm(p_client - p_server).item() ** 2
                total_drift += np.sqrt(drift)
        return total_drift / len(selected_indices)

    def exec_client_round(self, server: SmallCNN, clients: List[SmallCNN],
                         client_dataloaders: List[DataLoader], **kwargs):
        device = kwargs['device']
        lr = kwargs['lr']
        verbose = kwargs.get('verbose', False)
        current_round = kwargs.get('current_round', 0)
        criterion = nn.CrossEntropyLoss()
        n_clients = len(clients)
        selected_indices = self._sample_clients(n_clients, seed=current_round)
        print(f"Selected {len(selected_indices)}/{n_clients} clients: {selected_indices}")
        client_sizes = []
        for idx in selected_indices:
            client = clients[idx]
            loader = client_dataloaders[idx]
            print(f"Training Client {idx+1}/{n_clients} for {self.local_epochs} epochs")
            client.load_state_dict(server.state_dict())
            n_samples = self._train_client_local(client, loader, criterion, lr, device)
            client_sizes.append(n_samples)
            if verbose:
                print(f"  Client {idx+1}: trained on {n_samples} samples")
        drift = self._compute_client_drift(clients, server, selected_indices)
        self.round_metrics['client_drift'].append(drift)
        if verbose:
            print(f"Average client drift: {drift:.6f}")
        kwargs['client_sizes'] = client_sizes
        kwargs['selected_indices'] = selected_indices

    def exec_server_round(self, clients: List[SmallCNN], server: SmallCNN, **kwargs):
        verbose = kwargs.get('verbose', False)
        client_sizes = kwargs.get('client_sizes', None)
        selected_indices = kwargs.get('selected_indices', list(range(len(clients))))
        n_selected = len(selected_indices)
        if client_sizes is not None:
            weights = self._normalize_weights(n_selected, client_sizes)
        else:
            weights = [1.0 / n_selected] * n_selected
        print(f"Aggregating {n_selected} clients")
        if verbose:
            print(f"Weights: {[f'{w:.3f}' for w in weights]}")
        with torch.no_grad():
            for p_idx, server_param in enumerate(server.parameters()):
                aggregated = None
                for client_idx, weight in zip(selected_indices, weights):
                    client_param = list(clients[client_idx].parameters())[p_idx]
                    if client_param.device != server_param.device:
                        client_param = client_param.to(server_param.device)
                    weighted_param = client_param * weight
                    aggregated = weighted_param.clone() if aggregated is None else aggregated + weighted_param
                server_param.copy_(aggregated)

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
        print(f"FedAvg  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")
        print(f"Central | Test Loss: {central_loss:.4f}, Test Acc: {central_acc*100:.2f}%")
        print(f"Param Difference: {param_diff:.6e}")

    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        self.evaluate_round(server, central, **kwargs)

    def get_metrics(self):
        return self.round_metrics

class FedProx(FedMethod):
    """
        Implementation of the FedProx algorithm (Li et al., MLSys 2020).

        FedProx extends FedAvg by adding a proximal term to each client's local
        objective, penalizing deviation from the global model parameters. This
        stabilizes training and improves convergence under heterogeneous (non-IID)
        client data distributions. When μ = 0, FedProx reduces to FedAvg.
    """

    def __init__(self, client_weights: Optional[Sequence[float]] = None, local_rounds: int = 5):
        super().__init__()
        self.client_weights = None if client_weights is None else list(client_weights)
        self.K = local_rounds

        # Track metrics
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
        verbose = kwargs.get('verbose', False)
        device = kwargs.get('device', None)

        if n_clients == 0:
            raise ValueError("clients must contain at least one model.")
        
        # normalize weights
        weights = self._normalize_weights(n_clients)

        print("Applying FedProx on Server (parameter averaging)")
        if verbose:
            print(f"Aggregating {n_clients} clients with weights: {[f'{w:.3f}' for w in weights]}")

        if device is None:
            try:
                server_device = next(server.parameters()).device
            except StopIteration:
                server_device = torch.device('cpu')
        else:
            server_device = device

        with torch.no_grad():
            server_sd_before = copy.deepcopy({k: v.detach().clone().to(server_device)
                            for k, v in server.state_dict().items() if torch.is_tensor(v)})

            averaged_sd = {}
            for key, server_val in server.state_dict().items():
                # If value isn't a tensor, just copy server's
                if not torch.is_tensor(server_val):
                    averaged_sd[key] = server_val
                    continue
            
                agg = None
                for client, w in zip(clients, weights):
                    client_sd = client.state_dict()
                    client_val = client_sd[key]
                    if not torch.is_tensor(client_val):
                        # if client doesn't have a tensor at this key -> fallback to server_val
                        agg = server_val.detach().clone().to(server_device)
                        break

                    # Move client's tensor to server device for accumulation
                    t = client_val.detach().to(server_device)
                    weighted = t.mul(w)  # same dtype/device as t
                    agg = weighted.clone() if agg is None else agg.add_(weighted)

                # If no clients contributed (shouldn't happen), fallback to server's current value
                if agg is None:
                    agg = server_val.detach().clone().to(server_device)

                averaged_sd[key] = agg

            # Load averaged parameters/buffers into the server
            # ensure server is on server_device
            server.to(server_device)
            server.load_state_dict(averaged_sd)

            # Debug: compute total L2 change (||theta_new - theta_old||) and norm of new params
            total_change_sq = 0.0
            total_norm_sq = 0.0
            for k, new_val in averaged_sd.items():
                if not torch.is_tensor(new_val):
                    continue
                old_val = server_sd_before.get(k)
                if old_val is None:
                    continue
                diff = new_val - old_val
                total_change_sq += float(torch.norm(diff).pow(2).item())
                total_norm_sq += float(torch.norm(new_val).pow(2).item())

            total_change = total_change_sq ** 0.5
            agg_norm = total_norm_sq ** 0.5

            if verbose:
                print(f"  Server parameter L2-change: {total_change:.6f}")
                print(f"  Aggregated parameter L2-norm: {agg_norm:.6f}")

    def _train_client(self, server: SmallCNN, client: SmallCNN, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, device, **kwargs):
        client.to(device)
        client.train()

        learning_rate = kwargs.get('lr', 0.001)
        momentum = kwargs.get('momentum', 0)
        weight_decay = kwargs.get('weight_decay', 0)
        rounds = kwargs.get('rounds', self.K)

        mu = kwargs.get('mu', 0.01)
        theta_global = [p.detach().clone().to(device) for p in server.parameters()]

        optimizer = torch.optim.SGD(client.parameters(), lr=learning_rate, 
            momentum=momentum, 
            weight_decay=weight_decay)

        num_steps = 0
        counter = 0
        batch_steps = max(1, int(len(dataloader)/ self.K))
        leftover = len(dataloader) - batch_steps * self.K
        curr_samples = 0.0

        optimizer.zero_grad(set_to_none=True)
            
        total_samples = 0.0
        total_loss= 0.0

        for batch in dataloader:
            counter +=1

            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            out = client(inputs)
            loss = criterion(out, targets)
            
            batch_size = inputs.size(0)
            scaled_loss = loss * batch_size
            scaled_loss.backward()

            total_loss += loss.item() * batch_size
            curr_samples += inputs.size(0)
            total_samples += inputs.size(0)
                                                    
            if counter == (batch_steps + (1 if num_steps < leftover else 0)):
                
                if curr_samples > 0:
                    with torch.no_grad():
                        for p in client.parameters():
                            if p.grad is not None:
                                p.grad.div_(curr_samples)
                        # Prox Term
                        for p, p_g in zip(client.parameters(), theta_global):
                            prox = mu * (p.detach() - p_g)
                            if p.grad is None:
                                p.grad = prox.clone()
                            else:
                                p.grad.add_(prox)

                    curr_samples = 0.0

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                counter = 0
                num_steps += 1
                if leftover > 0: leftover -= 1

        if num_steps < rounds:
            if curr_samples > 0:
                with torch.no_grad():
                    for p in client.parameters():
                        if p.grad is not None:
                            p.grad.div_(curr_samples)
                    # Prox Term
                    for p, p_g in zip(client.parameters(), theta_global):
                        prox = mu * (p.detach() - p_g)
                        if p.grad is None:
                            p.grad = prox.clone()
                        else:
                            p.grad.add_(prox)
                curr_samples = 0.0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if total_samples == 0:
            return 0, 0.0

        avg_loss = total_loss / total_samples

        return total_samples, avg_loss

    def exec_client_round(self, server: SmallCNN, clients: List[SmallCNN], client_dataloaders: List[DataLoader], **kwargs):

        device = kwargs['device']
        verbose = kwargs['verbose']
        rounds = kwargs['rounds']

        criterion = nn.CrossEntropyLoss()

        client_sizes = []

        for i, (client, loader) in enumerate(zip(clients, client_dataloaders)):
            print(f"Training Client {i+1}/{len(clients)}")

            client.load_state_dict(copy.deepcopy(server.state_dict()))
            n_samples, avg_loss = self._train_client(server, client, loader, criterion, device, **kwargs)

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

        self.round_metrics['fed_test_acc'].append(server_acc)
        self.round_metrics['fed_test_loss'].append(server_loss)
        self.round_metrics['central_test_acc'].append(central_acc)
        self.round_metrics['central_test_loss'].append(central_loss)
        param_diff = compute_model_difference(server, central, norm_type='l2')
        self.round_metrics['param_difference'].append(param_diff)

        print(f"FedProx  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")
        print(f"Central | Test Loss: {central_loss:.4f}, Test Acc: {central_acc*100:.2f}%")

    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        self.evaluate_round(server, central, **kwargs)


