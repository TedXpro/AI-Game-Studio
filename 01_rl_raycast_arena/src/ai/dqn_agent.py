"""Double Deep Q-Network agent with epsilon schedule, Polyak sync, and checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from src.ai.q_network import QNetwork
from src.ai.replay_buffer import ReplayBuffer, TransitionBatch


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters governing Double-DQN exploration, optimization, and memory."""

    obs_dim: int = 133
    action_dim: int = 6
    hidden_dim: int = 160
    learning_rate: float = 8e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 60_000
    warmup_steps: int = 1_200
    train_frequency: int = 4
    tau: float = 0.015
    max_grad_norm: float = 5.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay_steps: int = 40_000


def compute_double_dqn_targets(
    online_net: QNetwork,
    target_net: QNetwork,
    batch: TransitionBatch,
    gamma: float,
) -> torch.Tensor:
    """Computes decoupled Double-DQN Bellman targets without gradient tracking.

    Args:
        online_net: Active policy network selecting greedy next-state actions.
        target_net: Slowly-updated target network evaluating chosen actions.
        batch: Sampled off-policy transition mini-batch.
        gamma: Discount factor in [0.0, 1.0].

    Returns:
        Target Q-value tensor of shape (batch_size,).
    """
    with torch.no_grad():
        next_online_q = online_net(batch.next_states)
        best_next_actions = next_online_q.argmax(dim=1, keepdim=True)
        next_target_q = target_net(batch.next_states).gather(1, best_next_actions).squeeze(1)
        return batch.rewards + gamma * (1.0 - batch.dones) * next_target_q


class DQNAgent:
    """Autonomous Double-DQN decision maker and off-policy learner."""

    def __init__(
        self,
        config: DQNConfig | None = None,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        """Initializes online/target networks, optimizer, and replay memory.

        Args:
            config: Hyperparameter bundle; uses defaults when None.
            seed: Seed for reproducible exploration and network weights.
            device: Optional device string ('cpu' or 'cuda'); auto-detected when None.
        """
        self.config = config if config is not None else DQNConfig()
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        self.online_net = QNetwork(
            self.config.obs_dim, self.config.action_dim, self.config.hidden_dim
        ).to(self.device)
        self.target_net = QNetwork(
            self.config.obs_dim, self.config.action_dim, self.config.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(), lr=self.config.learning_rate, weight_decay=1e-5
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_capacity, self.config.obs_dim, seed=seed
        )
        self.total_steps = 0
        self.last_loss = 0.0

    @property
    def epsilon(self) -> float:
        """Returns the current exploration probability under linear annealing."""
        progress = min(1.0, self.total_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(
        self,
        state: NDArray[np.float32],
        greedy: bool = False,
    ) -> tuple[int, NDArray[np.float32]]:
        """Chooses an action via epsilon-greedy policy and returns live Q-values.

        Args:
            state: Current flat observation vector.
            greedy: When True, bypasses epsilon exploration and picks argmax Q(s, a).

        Returns:
            Tuple of (chosen_action_index, q_values_array).
        """
        q_values = self.online_net.predict_single(state, self.device)
        if (not greedy) and (float(self._rng.random()) < self.epsilon):
            action = int(self._rng.integers(0, self.config.action_dim))
        else:
            action = int(np.argmax(q_values))
        return action, q_values

    def record_step(
        self,
        state: NDArray[np.float32],
        action: int,
        reward: float,
        next_state: NDArray[np.float32],
        terminated: bool,
    ) -> float | None:
        """Stores a transition and triggers a gradient update when scheduled.

        Args:
            state: Pre-action observation vector.
            action: Executed action index.
            reward: Observed scalar reward.
            next_state: Post-action observation vector.
            terminated: True only when episode ended via crash.

        Returns:
            Scalar Huber loss if an optimization step ran, else None.
        """
        self.replay_buffer.push(state, action, reward, next_state, terminated)
        self.total_steps += 1
        ready = len(self.replay_buffer) >= max(self.config.warmup_steps, self.config.batch_size)
        if ready and (self.total_steps % self.config.train_frequency == 0):
            return self.optimize_step()
        return None

    def optimize_step(self, batch: TransitionBatch | None = None) -> float:
        """Executes one Double-DQN gradient step and soft-updates the target network.

        Args:
            batch: Optional explicit transition batch; samples from replay memory if None.

        Returns:
            Scalar Huber loss value for telemetry tracking.
        """
        active_batch = (
            batch
            if batch is not None
            else self.replay_buffer.sample(self.config.batch_size, self.device)
        )
        predicted_q = (
            self.online_net(active_batch.states)
            .gather(1, active_batch.actions.unsqueeze(1))
            .squeeze(1)
        )
        targets = compute_double_dqn_targets(
            self.online_net, self.target_net, active_batch, self.config.gamma
        )

        loss = self.loss_fn(predicted_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self._soft_update_target()

        self.last_loss = float(loss.item())
        return self.last_loss

    def _soft_update_target(self) -> None:
        """Blends online parameters into target parameters using Polyak averaging."""
        tau = self.config.tau
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_net.parameters(), self.online_net.parameters(), strict=True
            ):
                target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    def save_checkpoint(self, path: str | Path) -> None:
        """Serializes online/target network weights and step counter to disk.

        Args:
            path: Destination file path (.pt).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "total_steps": int(self.total_steps),
        }
        torch.save(payload, dest)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restores online/target network weights and step counter from disk.

        Args:
            path: Source checkpoint file path (.pt).
        """
        payload = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(payload["online_state_dict"])
        self.target_net.load_state_dict(payload["target_state_dict"])
        self.total_steps = int(payload.get("total_steps", self.config.epsilon_decay_steps))
