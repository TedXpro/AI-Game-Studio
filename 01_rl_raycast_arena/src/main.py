"""Command-line and interactive telemetry entrypoint for Raycast Survival Arena."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from src.ai.dqn_agent import DQNAgent, DQNConfig
from src.sim.arena_env import RaycastArenaEnv
from src.sim.entities import ArenaConfig


DEFAULT_CHECKPOINT = Path("checkpoints/best_dqn.pt")


def parse_args() -> argparse.Namespace:
    """Parses command-line execution flags for headless or interactive modes.

    Returns:
        Populated argparse Namespace.
    """
    parser = argparse.ArgumentParser(description="01 - Raycast Survival Arena (Double DQN)")
    parser.add_argument("--headless", action="store_true", help="Run turbo training without GUI.")
    parser.add_argument("--episodes", type=int, default=450, help="Target headless episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--load", type=str, default="", help="Optional checkpoint path to load.")
    parser.add_argument("--eval", action="store_true", help="Start in greedy evaluation mode.")
    return parser.parse_args()


def evaluate_greedy_policy(
    env: RaycastArenaEnv,
    agent: DQNAgent,
    num_episodes: int = 20,
    base_seed: int = 10_000,
) -> tuple[float, float]:
    """Benchmarks the current greedy policy without exploration noise.

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


def run_headless_training(
    env: RaycastArenaEnv,
    agent: DQNAgent,
    episodes: int,
    checkpoint_path: Path,
) -> None:
    """Executes maximum-throughput CPU/GPU training and saves the best checkpoint.

    Args:
        env: Headless simulation environment.
        agent: Double-DQN agent to train.
        episodes: Total training episodes to execute.
        checkpoint_path: Destination file path for the highest-scoring model.
    """
    eval_env = RaycastArenaEnv(env.config, seed=999)
    reward_history: list[float] = []
    core_history: list[int] = []
    best_eval_reward = -float("inf")
    train_seconds = 0.0

    init_rwd, init_cores = evaluate_greedy_policy(eval_env, agent, num_episodes=10)
    print(
        f"[Untrained Baseline] Greedy Avg Reward: {init_rwd:+7.2f} | "
        f"Avg Cores: {init_cores:4.2f} | Device: {agent.device}"
    )

    for ep in range(1, episodes + 1):
        t_ep = time.perf_counter()
        state = env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(state, greedy=False)
            result = env.step(action)
            agent.record_step(
                state, action, result.reward, result.observation, result.terminated
            )
            state = result.observation
            done = result.terminated or result.truncated
        train_seconds += time.perf_counter() - t_ep

        reward_history.append(env.episode_reward)
        core_history.append(env.episode_cores)

        if ep % 50 == 0 or ep == episodes:
            sps = agent.total_steps / max(1e-6, train_seconds)
            avg_r = float(np.mean(reward_history[-50:]))
            avg_c = float(np.mean(core_history[-50:]))
            eval_r, eval_c = evaluate_greedy_policy(eval_env, agent, num_episodes=10)
            marker = ""
            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                agent.save_checkpoint(checkpoint_path)
                marker = " *saved*"
            print(
                f"Ep {ep:4d}/{episodes} | Eps {agent.epsilon:.3f} | "
                f"Train50 {avg_r:+7.2f} ({avg_c:4.2f} cores) | "
                f"Eval {eval_r:+7.2f} ({eval_c:4.2f} cores) | "
                f"{sps:5.0f} st/s{marker}"
            )


def run_interactive_lab(
    env: RaycastArenaEnv,
    agent: DQNAgent,
    checkpoint_path: Path,
    start_eval: bool = False,
) -> None:
    """Runs the 60 FPS / Turbo-toggleable live Pygame-CE telemetry laboratory.

    Args:
        env: Arena environment instance.
        agent: Double-DQN agent driving the scout drone.
        checkpoint_path: Path used by quick-save and quick-load hotkeys.
        start_eval: Whether to begin in greedy evaluation mode.
    """
    import pygame
    from src.view.visualizer import ArenaVisualizer, TelemetryFrame

    visualizer = ArenaVisualizer(env.config, offscreen=False)
    turbo_mode = False
    eval_mode = start_eval
    show_rays = True
    episode = 1
    reward_history: list[float] = []
    best_reward = -float("inf")
    state = env.reset()
    last_action = 0
    last_q = np.zeros(env.config.action_dim, dtype=np.float32)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    turbo_mode = not turbo_mode
                elif event.key == pygame.K_e:
                    eval_mode = not eval_mode
                elif event.key == pygame.K_r:
                    show_rays = not show_rays
                elif event.key == pygame.K_s:
                    agent.save_checkpoint(checkpoint_path)
                elif event.key == pygame.K_l and checkpoint_path.exists():
                    agent.load_checkpoint(checkpoint_path)

        steps_this_frame = 240 if (turbo_mode and not eval_mode) else 1
        t0 = time.perf_counter()
        for _ in range(steps_this_frame):
            last_action, last_q = agent.select_action(state, greedy=eval_mode)
            res = env.step(last_action)
            if not eval_mode:
                agent.record_step(
                    state, last_action, res.reward, res.observation, res.terminated
                )
            state = res.observation
            if res.terminated or res.truncated:
                reward_history.append(env.episode_reward)
                best_reward = max(best_reward, env.episode_reward)
                episode += 1
                state = env.reset()

        dt = max(1e-4, time.perf_counter() - t0)
        sps = steps_this_frame / dt if turbo_mode else 60.0
        avg_r = float(np.mean(reward_history[-50:])) if reward_history else 0.0
        mode = "GREEDY EVAL" if eval_mode else ("TURBO TRAIN" if turbo_mode else "LIVE 60 FPS")
        frame = TelemetryFrame(
            episode=episode,
            epsilon=0.0 if eval_mode else agent.epsilon,
            avg_reward=avg_r,
            best_reward=0.0 if best_reward == -float("inf") else best_reward,
            steps_per_sec=sps,
            q_values=last_q,
            chosen_action=last_action,
            reward_history=reward_history,
            mode_label=mode,
            show_rays=show_rays,
        )
        visualizer.render(env.get_snapshot(), frame, fps_limit=0 if turbo_mode else 60)

    visualizer.close()


def main() -> None:
    """Entrypoint wiring CLI arguments to headless training or interactive telemetry."""
    args = parse_args()
    arena_cfg = ArenaConfig()
    dqn_cfg = DQNConfig(obs_dim=arena_cfg.observation_dim, action_dim=arena_cfg.action_dim)
    env = RaycastArenaEnv(arena_cfg, seed=args.seed)
    agent = DQNAgent(dqn_cfg, seed=args.seed)

    ckpt = Path(args.load) if args.load else DEFAULT_CHECKPOINT
    if args.load and ckpt.exists():
        agent.load_checkpoint(ckpt)

    if args.headless:
        run_headless_training(env, agent, episodes=args.episodes, checkpoint_path=ckpt)
    else:
        run_interactive_lab(env, agent, checkpoint_path=ckpt, start_eval=args.eval)


if __name__ == "__main__":
    main()
