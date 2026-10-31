"""Headless training loop, evolutionary milestone saver, and JSON telemetry logger."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
import numpy as np

from src.ai.dqn_agent import DQNAgent
from src.sim.arena_env import RaycastArenaEnv
from src.sim.entities import is_wall_collision


MILESTONE_SPECS: dict[int, tuple[str, str]] = {
    1: ("milestone_1_ep000.pt", "STAGE 1: EP 0 UNTRAINED"),
    2: ("milestone_2_ep035.pt", "STAGE 2: EP 35 NOVICE"),
    3: ("milestone_3_ep090.pt", "STAGE 3: EP 90 SKILLED"),
    4: ("best_dqn.pt", "STAGE 4: CHAMPION"),
}

HISTORY_FILENAME: str = "training_history.json"


@dataclass
class TrainingHistory:
    """Episode-by-episode RL progression metrics serialized for plotting and analysis."""

    episodes: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    cores: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    mean_q: list[float] = field(default_factory=list)
    mean_loss: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    eval_episodes: list[int] = field(default_factory=list)
    eval_rewards: list[float] = field(default_factory=list)
    eval_cores: list[float] = field(default_factory=list)

    def save_json(self, path: str | Path) -> None:
        """Writes all recorded training arrays to a formatted JSON file.

        Args:
            path: Target JSON file path.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> TrainingHistory:
        """Restores a TrainingHistory instance from a saved JSON file.

        Args:
            path: Source JSON file path.

        Returns:
            Populated TrainingHistory dataclass.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


def evaluate_greedy_policy(
    env: RaycastArenaEnv,
    agent: DQNAgent,
    num_episodes: int = 12,
    base_seed: int = 10_000,
) -> tuple[float, float]:
    """Benchmarks the greedy policy without exploration noise across seeded arenas.

    Args:
        env: Headless arena instance used for evaluation rollouts.
        agent: Trained or partially trained DQNAgent.
        num_episodes: Number of deterministic evaluation episodes.
        base_seed: Starting seed offset for reproducible test layouts.

    Returns:
        Tuple of (mean_episode_reward, mean_cores_collected).
    """
    rewards: list[float] = []
    cores: list[int] = []
    for ep in range(num_episodes):
        state = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            action, _ = agent.select_action(state, greedy=True)
            step_res = env.step(action)
            state = step_res.observation
            done = step_res.terminated or step_res.truncated
        rewards.append(env.episode_reward)
        cores.append(env.episode_cores)
    return float(np.mean(rewards)), float(np.mean(cores))


def resolve_episode_outcome(env: RaycastArenaEnv, terminated: bool) -> str:
    """Classifies how an episode ended into wall crash, mine crash, or survival.

    Args:
        env: Arena environment at episode end.
        terminated: Whether the episode ended via collision.

    Returns:
        Outcome label in {'wall', 'mine', 'survived'}.
    """
    if not terminated:
        return "survived"
    return "wall" if is_wall_collision(env.agent, env.config) else "mine"


def run_training_session(
    env: RaycastArenaEnv,
    agent: DQNAgent,
    episodes: int,
    checkpoint_dir: str | Path,
    eval_interval: int = 25,
    eval_episodes: int = 12,
    verbose: bool = True,
) -> TrainingHistory:
    """Trains the agent, saves Stages 1-4 milestones, and logs JSON progression metrics.

    Args:
        env: Headless simulation environment.
        agent: Double-DQN agent to train.
        episodes: Total training episodes to execute.
        checkpoint_dir: Directory storing milestone .pt files and training_history.json.
        eval_interval: Episode cadence for deterministic greedy evaluations.
        eval_episodes: Number of unseen seeds per evaluation checkpoint.
        verbose: Whether to print periodic telemetry rows to stdout.

    Returns:
        Complete TrainingHistory object across all episodes.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = TrainingHistory()
    eval_env = RaycastArenaEnv(env.config, seed=999)

    stage2_ep = 35 if episodes >= 120 else max(1, episodes // 3)
    stage3_ep = 90 if episodes >= 120 else max(2, (2 * episodes) // 3)

    agent.save_checkpoint(ckpt_dir / MILESTONE_SPECS[1][0])
    best_path = ckpt_dir / MILESTONE_SPECS[4][0]
    agent.save_checkpoint(best_path)

    init_r, init_c = evaluate_greedy_policy(eval_env, agent, num_episodes=eval_episodes)
    history.eval_episodes.append(0)
    history.eval_rewards.append(round(init_r, 3))
    history.eval_cores.append(round(init_c, 3))
    best_eval_reward = init_r
    train_seconds = 0.0

    if verbose:
        print(
            f"[Stage 1 Saved | Ep 0] Greedy Eval: {init_r:+7.2f} "
            f"({init_c:4.2f} cores) | Device: {agent.device}"
        )

    for ep in range(1, episodes + 1):
        t0 = time.perf_counter()
        state = env.reset()
        done = False
        terminated = False
        q_maxes: list[float] = []
        losses: list[float] = []

        while not done:
            action, q_vals = agent.select_action(state, greedy=False)
            q_maxes.append(float(np.max(q_vals)))
            res = env.step(action)
            loss = agent.record_step(
                state, action, res.reward, res.observation, res.terminated
            )
            if loss is not None:
                losses.append(loss)
            state = res.observation
            terminated = res.terminated
            done = res.terminated or res.truncated

        train_seconds += time.perf_counter() - t0
        history.episodes.append(ep)
        history.rewards.append(round(env.episode_reward, 3))
        history.cores.append(int(env.episode_cores))
        history.steps.append(int(env.step_count))
        history.outcomes.append(resolve_episode_outcome(env, terminated))
        history.mean_q.append(round(float(np.mean(q_maxes)), 3))
        history.mean_loss.append(round(float(np.mean(losses)) if losses else 0.0, 4))
        history.epsilons.append(round(agent.epsilon, 4))

        if ep == stage2_ep:
            agent.save_checkpoint(ckpt_dir / MILESTONE_SPECS[2][0])
        if ep == stage3_ep:
            agent.save_checkpoint(ckpt_dir / MILESTONE_SPECS[3][0])

        if ep % eval_interval == 0 or ep == episodes:
            eval_r, eval_c = evaluate_greedy_policy(eval_env, agent, num_episodes=eval_episodes)
            history.eval_episodes.append(ep)
            history.eval_rewards.append(round(eval_r, 3))
            history.eval_cores.append(round(eval_c, 3))
            marker = ""
            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                agent.save_checkpoint(best_path)
                marker = " *champion*"
            if verbose:
                sps = agent.total_steps / max(1e-6, train_seconds)
                avg_r = float(np.mean(history.rewards[-50:]))
                avg_c = float(np.mean(history.cores[-50:]))
                print(
                    f"Ep {ep:4d}/{episodes} | Eps {agent.epsilon:.3f} | "
                    f"Train50 {avg_r:+7.2f} ({avg_c:4.2f}c) | "
                    f"Eval {eval_r:+7.2f} ({eval_c:5.2f}c) | {sps:5.0f} st/s{marker}"
                )

    history.save_json(ckpt_dir / HISTORY_FILENAME)
    return history
