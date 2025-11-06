import torch
import torch.nn as nn
from typing import List, Optional, Sequence, Dict
from utils import (
    SmallCNN,
    train_model_one_epoch,
)

class FedMethod:
    """
    Abstract Class for Fed Methods
    """
    def __init__(self):
        pass

    def exec_round(self, clients: List[SmallCNN], server: SmallCNN):
        pass

class FedSGD(FedMethod):
    """
    Federated SGD aggregator.

    Args:
        client_weights (Optional[Sequence[float]]): optional per-client weights, for e.g. domain representation ratio: N_i / N or simply N_i. If None,
            equal weights are used.
    """
    def __init__(self, client_weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.client_weights = None if client_weights is None else list(client_weights)

    def _normalize_weights(self, n_clients: int) -> List[float]:
        if self.client_weights is None:
            return [1.0 / n_clients] * n_clients
        if len(self.client_weights) != n_clients:
            raise ValueError("Length of client_weights must equal number of clients.")
        w = [float(x) for x in self.client_weights]
        s = sum(w)
        if s <= 0.0:
            raise ValueError("Sum of client_weights must be positive.")
        return [x / s for x in w]

    def exec_round(self, clients: List[SmallCNN], server: SmallCNN):
        """
        Execute one federated SGD round: average client gradients (weighted) and update server params.

        Args:
            clients: list of SmallCNNs trained for one epoch.
            server: the server model (nn.Module) whose parameters will be updated.

        Returns:
            dict with keys:
              - "weighted_loss": float weighted average of client losses (detached)
              - "grad_norm": float L2 norm of aggregated gradients (after averaging)
        """
        if not isinstance(clients, (list, tuple)):
            raise TypeError("clients must be a list/tuple of SmallCNN Models.")

        n_clients = len(clients)
        if n_clients == 0:
            raise ValueError("clients must contain at least one Model.")
        
        # normalize weights
        weights = self._normalize_weights(n_clients)

        with torch.no_grad():
            for server_param, *client_params in zip(server.parameters(), *[c.parameters() for c in clients]):
                server_param.data.copy_(sum(w * p for w,p in zip(weights, client_params)))