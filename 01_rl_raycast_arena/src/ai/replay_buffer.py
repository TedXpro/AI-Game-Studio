"""Pre-allocated circular experience replay buffer for off-policy RL training."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import torch


@dataclass(frozen=True)
class TransitionBatch:
    """Mini-batch of off-policy MDP transitions residing on a target Torch device."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity ring buffer backed by contiguous NumPy arrays."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 42) -> None:
        """Allocates contiguous memory blocks for experience replay.

        Args:
            capacity: Maximum number of stored transitions before overwriting oldest.
            obs_dim: Length of each observation vector.
            seed: Random seed governing uniform mini-batch sampling.
        """
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._rng = np.random.default_rng(seed)

        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        """Returns the current number of valid transitions stored in memory."""
        return self._size

    def push(
        self,
        state: NDArray[np.float32],
        action: int,
        reward: float,
        next_state: NDArray[np.float32],
        terminated: bool,
    ) -> None:
        """Inserts a single transition into the circular replay buffer.

        Args:
            state: Observation prior to action of shape (obs_dim,).
            action: Discrete action index executed.
            reward: Scalar reward observed.
            next_state: Resulting observation of shape (obs_dim,).
            terminated: True only on true terminal states (crashes), not time limits.
        """
        idx = self._cursor
        self._states[idx] = state
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = next_state
        self._dones[idx] = 1.0 if terminated else 0.0

        self._cursor = (idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        """Draws a uniform random mini-batch of transitions onto the target Torch device.

        Args:
            batch_size: Number of transitions to sample.
            device: Destination PyTorch device for tensor placement.

        Returns:
            TransitionBatch ready for Bellman target computation.
        """
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} transitions from size {self._size}.")

        indices = self._rng.integers(0, self._size, size=batch_size)
        return TransitionBatch(
            states=torch.as_tensor(self._states[indices], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self._actions[indices], dtype=torch.int64, device=device),
            rewards=torch.as_tensor(self._rewards[indices], dtype=torch.float32, device=device),
            next_states=torch.as_tensor(
                self._next_states[indices], dtype=torch.float32, device=device
            ),
            dones=torch.as_tensor(self._dones[indices], dtype=torch.float32, device=device),
        )
