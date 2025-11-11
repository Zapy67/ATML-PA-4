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
        verbose = kwargs['verbose']

        if n_clients == 0:
            raise ValueError("clients must contain at least one model.")
        
        # normalize weights
        weights = self._normalize_weights(n_clients)

        server_optimizer: torch.optim.SGD = kwargs['server_optimizer']

        print("Applying FedSGD on Server")
        if verbose:
            print(f"Aggregating {n_clients} clients with weights: {[f'{w:.3f}' for w in weights]}")

        server_optimizer.zero_grad(set_to_none=True)

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


class FedSAM(FedMethod):
    def __init__(self, 
             sam_rho: float, 
             num_local_steps: int, 
             client_aggregation_weights: Optional[Sequence[float]] = None):
    
        super().__init__()
        self.client_weights = None if client_aggregation_weights is None else list(client_aggregation_weights)
        self.rho = sam_rho
        self.K = num_local_steps

    
    def debug_output(self, model):
        with torch.no_grad:
            params = [param.detach().flatten() for param in model.parameters() 
                                        if param is not None]
            if params:
                param_vector = torch.cat(params)
                print(f"Aggregated server grad_norm: {torch.norm(param_vector).item():.4f}")
    
    def exec_server_round(self, 
                        local_models: List[SmallCNN], 
                        global_model: SmallCNN, 
                        **kwargs):
        """
        Execute one federated SAM round: average client gradients (weighted) and update server params.

        Args:
            local_models: list of SmallCNNs trained for one epoch.
            global_model: the server model (nn.Module) whose parameters will be updated.
        """

        num_clients = len(local_models)
        verbose = kwargs['verbose']
    
        if num_clients == 0:
            raise ValueError("local_models must contain at least one model.")
        
        aggregation_weights = self.client_weights

        print("Applying FedSAM on Server")
        if verbose:
            print(f"Aggregating {num_clients} clients with weights: {[f'{weight:.3f}' for weight in aggregation_weights]}")

        with torch.no_grad():
            
            param_groups = zip(*[list(local_model.parameters()) for local_model in local_models])
            
            aggregated_params = [agg_weight * torch.mean(torch.stack(param_group), dim=0) 
                                for param_group, agg_weight in zip(param_groups, aggregation_weights)]
            
            for aggregated_param, global_param in zip(aggregated_params, global_model.parameters()):
                global_param.data = aggregated_param
                
        if verbose:
            self.debug_output(global_model)

    def _train_client(self, 
                  local_model: SmallCNN, 
                  local_dataloader: DataLoader,
                  device: torch.device, 
                  criterion: nn.CrossEntropyLoss,
                  **kwargs):
    
        local_model.to(device)
        local_model.train()

        learning_rate = kwargs.get('lr', 0.5 * self.rho)
        momentum = kwargs.get('momentum', 0)
        weight_decay = kwargs.get('weight_decay', 0)

        total_samples_processed = 0.0
        total_loss_accumulated = 0.0
        
     
        local_optimizer = torch.optim.SGD(
            local_model.parameters(), 
            lr=learning_rate, 
            momentum=momentum, 
            weight_decay=weight_decay
        )

        for batch_index, data_batch in enumerate(local_dataloader):
            if batch_index == self.K:
                break
            
            batch_inputs, batch_targets = data_batch
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            def calculate_gradients_closure():
                local_optimizer.zero_grad()
                predictions = local_model(batch_inputs) 
                loss = criterion(predictions, batch_targets)
                loss.backward()
                return loss
            
            original_model_state = local_model.state_dict()
            
            calculate_gradients_closure()
            
            for model_param in local_model.parameters():
                model_param.data += self.rho * model_param.grad / torch.norm(model_param.grad)
            
            calculate_gradients_closure()
            
            local_model.load_state_dict(original_model_state)
            
            local_optimizer.step()

            total_loss_accumulated += local_optimizer.step(calculate_gradients_closure).item()
            total_samples_processed += batch_inputs.size(0)
        
        if total_samples_processed == 0:
            return 0, 0.0
        
        average_loss = total_loss_accumulated / total_samples_processed

        return total_samples_processed, average_loss

    def exec_client_round(self, server: SmallCNN, clients: List[SmallCNN], client_dataloaders: List[DataLoader], **kwargs):

        device = kwargs['device']
        verbose = kwargs['verbose']

        criterion = nn.CrossEntropyLoss(reduction='sum')

        client_sizes = []
        client_losses = []

        for i, (client, loader) in enumerate(zip(clients, client_dataloaders)):
            print(f"Training Client {i+1}/{len(clients)}")

            client.load_state_dict(server.state_dict())
            n_samples, avg_loss = self._train_client(client, loader, criterion, device)

            client_sizes.append(n_samples)
            client_losses.append(avg_loss)

            if verbose:
                self.debug_output(client)

        kwargs['client_losses'] = client_losses
        kwargs['client_sizes'] = client_sizes


    def evaluate_round(self, server: SmallCNN, central: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss()
        device = kwargs['device']
        test_loader = kwargs['test_loader']

        server_loss, server_acc = evaluate_model_on_test(server, test_loader, criterion, device)
        central_loss, central_acc = evaluate_model_on_test(central, test_loader, criterion, device)

        print(f"FedSAM  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")
        print(f"Central | Test Loss: {central_loss:.4f}, Test Acc: {central_acc*100:.2f}%")

    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        self.evaluate_round(server, central, **kwargs)



class FedGH(FedMethod):
    def __init__(self, 
             num_local_steps: int, 
             client_aggregation_weights: Optional[Sequence[float]] = None):
    
        super().__init__()
        self.client_weights = None if client_aggregation_weights is None else list(client_aggregation_weights)
        self.K = num_local_steps

    
    def debug_output(self, model):
        with torch.no_grad:
            params = [param.detach().flatten() for param in model.parameters() 
                                        if param is not None]
            if params:
                param_vector = torch.cat(params)
                print(f"Aggregated server grad_norm: {torch.norm(param_vector).item():.4f}")

    
    def harmonize_updates(self, 
                        local_models: List[SmallCNN], 
                        global_model: SmallCNN, 
                        ):
        # import random
        # flattened_client_params = [torch.cat([param.flatten() for param in local_model.parameters()]) for local_model in local_models]
        # global_flattened = torch.cat([param.flatten() for param in global_model.parameters()])
        
        # deltas = random.shuffle([global_flattened - stacked_params for stacked_params in flattened_client_params])

        # for idx, delta in enumerate(deltas):
        #     i = idx
        #     while i < len(deltas):
        #         i += 1
        #         if torch.dot(deltas[i], deltas[idx]) < 0:
        #             harmonized = deltas[i] - 
        pass
        

    
    def exec_server_round(self, 
                        local_models: List[SmallCNN], 
                        global_model: SmallCNN, 
                        **kwargs):
        """
        Execute one federated SAM round: average client gradients (weighted) and update server params.

        Args:
            local_models: list of SmallCNNs trained for one epoch.
            global_model: the server model (nn.Module) whose parameters will be updated.
        """

        num_clients = len(local_models)
        verbose = kwargs['verbose']
    
        if num_clients == 0:
            raise ValueError("local_models must contain at least one model.")
        
        aggregation_weights = self.client_weights

        print("Applying FedSAM on Server")
        if verbose:
            print(f"Aggregating {num_clients} clients with weights: {[f'{weight:.3f}' for weight in aggregation_weights]}")

        with torch.no_grad():
            
            param_groups = zip(*[list(local_model.parameters()) for local_model in local_models])
            
            aggregated_params = [agg_weight * torch.mean(torch.stack(param_group), dim=0) 
                                for param_group, agg_weight in zip(param_groups, aggregation_weights)]
            
            for aggregated_param, global_param in zip(aggregated_params, global_model.parameters()):
                global_param.data = aggregated_param
                
        if verbose:
            self.debug_output(global_model)

    def _train_client(self, 
                  local_model: SmallCNN, 
                  local_dataloader: DataLoader,
                  device: torch.device, 
                  criterion: nn.CrossEntropyLoss,
                  **kwargs):
    
        local_model.to(device)
        local_model.train()

        learning_rate = kwargs.get('lr', 0.5 * self.rho)
        momentum = kwargs.get('momentum', 0)
        weight_decay = kwargs.get('weight_decay', 0)

        total_samples_processed = 0.0
        total_loss_accumulated = 0.0
        
     
        local_optimizer = torch.optim.SGD(
            local_model.parameters(), 
            lr=learning_rate, 
            momentum=momentum, 
            weight_decay=weight_decay
        )

        for batch_index, data_batch in enumerate(local_dataloader):
            if batch_index == self.K:
                break
            
            batch_inputs, batch_targets = data_batch
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

           
            local_optimizer.zero_grad()
            predictions = local_model(batch_inputs) 
            loss = criterion(predictions, batch_targets)
            loss.backward()
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

        criterion = nn.CrossEntropyLoss(reduction='sum')

        client_sizes = []
        client_losses = []

        for i, (client, loader) in enumerate(zip(clients, client_dataloaders)):
            print(f"Training Client {i+1}/{len(clients)}")

            client.load_state_dict(server.state_dict())
            n_samples, avg_loss = self._train_client(client, loader, criterion, device)

            client_sizes.append(n_samples)
            client_losses.append(avg_loss)

            if verbose:
                self.debug_output(client)

        kwargs['client_losses'] = client_losses
        kwargs['client_sizes'] = client_sizes


    def evaluate_round(self, server: SmallCNN, central: SmallCNN, **kwargs):
        criterion = nn.CrossEntropyLoss()
        device = kwargs['device']
        test_loader = kwargs['test_loader']

        server_loss, server_acc = evaluate_model_on_test(server, test_loader, criterion, device)
        central_loss, central_acc = evaluate_model_on_test(central, test_loader, criterion, device)

        print(f"FedSAM  | Test Loss: {server_loss:.4f}, Test Acc: {server_acc*100:.2f}%")
        print(f"Central | Test Loss: {central_loss:.4f}, Test Acc: {central_acc*100:.2f}%")

    def evaluate_server(self, server: SmallCNN, central: SmallCNN, **kwargs):
        self.evaluate_round(server, central, **kwargs)






