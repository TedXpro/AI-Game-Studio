"""PyTorch multilayer perceptron for state-action Q-value estimation."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn


class QNetwork(nn.Module):
    """Feedforward Q-network with LayerNorm stabilization and orthogonal initialization."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        """Constructs the MLP mapping egocentric observations to action Q-values.

        Args:
            obs_dim: Dimension of the flat observation vector.
            action_dim: Number of discrete actions.
            hidden_dim: Width of each hidden representation layer.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Applies orthogonal weight initialization to linear layers."""
        relu_gain = math.sqrt(2.0)
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=relu_gain)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Computes state-action Q-values for a batch of observation tensors.

        Args:
            states: Float tensor of shape (batch_size, obs_dim).

        Returns:
            Float tensor of Q-values of shape (batch_size, action_dim).
        """
        features = self.trunk(states)
        return self.head(features)

    @torch.no_grad()
    def predict_single(
        self,
        state: NDArray[np.float32],
        device: torch.device,
    ) -> NDArray[np.float32]:
        """Evaluates Q-values for a single NumPy observation without tracking gradients.

        Args:
            state: Flat float32 observation vector of shape (obs_dim,).
            device: Torch execution device hosting model parameters.

        Returns:
            NumPy float32 array of action Q-values of shape (action_dim,).
        """
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = self.forward(tensor).squeeze(0)
        return q_values.detach().cpu().numpy().astype(np.float32)
