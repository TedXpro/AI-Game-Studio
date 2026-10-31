"""Command-line and interactive telemetry entrypoint for Raycast Survival Arena."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import torch

from src.ai.dqn_agent import DQNAgent, DQNConfig
from src.ai.trainer import (
    HISTORY_FILENAME,
    MILESTONE_SPECS,
    TrainingHistory,
    run_training_session,
)
from src.sim.arena_env import RaycastArenaEnv
from src.sim.entities import ArenaConfig


DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


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
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=0,
        help="Optional evolutionary milestone stage (1-4) to launch immediately.",
    )
    return parser.parse_args()


def load_saved_reward_curve(checkpoint_dir: Path) -> list[float]:
    """Loads historical episode rewards from disk if available for HUD sparkline display.

    Args:
        checkpoint_dir: Directory containing training_history.json.

    Returns:
        List of historical episode rewards, or empty list if unavailable.
    """
    hist_path = checkpoint_dir / HISTORY_FILENAME
    if not hist_path.exists():
        return []
    return list(TrainingHistory.load_json(hist_path).rewards)


def try_activate_stage(
    stage_num: int,
    agent: DQNAgent,
    checkpoint_dir: Path,
) -> str | None:
    """Loads one of the four evolutionary milestone checkpoints if present on disk.

    Args:
        stage_num: Milestone index in {1, 2, 3, 4}.
        agent: DQNAgent whose weights will be updated.
        checkpoint_dir: Directory containing saved milestone .pt files.

    Returns:
        Stage display label on success, or None if the checkpoint file is missing.
    """
    filename, label = MILESTONE_SPECS[stage_num]
    target = checkpoint_dir / filename
    if not target.exists():
        return None
    agent.load_checkpoint(target)
    return label


def run_interactive_lab(
    env: RaycastArenaEnv,
    agent: DQNAgent,
    checkpoint_dir: Path,
    start_eval: bool = False,
    initial_stage_label: str = "",
) -> None:
    """Runs the 60 FPS / Turbo-toggleable live Pygame-CE telemetry laboratory.

    Args:
        env: Arena environment instance.
        agent: Double-DQN agent driving the scout drone.
        checkpoint_dir: Directory storing milestone .pt files and training history.
        start_eval: Whether to begin in endless greedy evaluation mode.
        initial_stage_label: Active milestone label displayed in the HUD header.
    """
    import pygame
    from src.view.visualizer import ArenaVisualizer, TelemetryFrame

    visualizer = ArenaVisualizer(env.config, offscreen=False)
    turbo_mode = False
    eval_mode = start_eval
    stage_label = initial_stage_label
    show_rays = True
    episode = 1
    reward_history = load_saved_reward_curve(checkpoint_dir) if start_eval else []
    best_reward = max(reward_history) if reward_history else -float("inf")
    state = env.reset()
    last_action = 0
    last_q = np.zeros(env.config.action_dim, dtype=np.float32)

    stage_keys = {
        pygame.K_1: 1,
        pygame.K_2: 2,
        pygame.K_3: 3,
        pygame.K_4: 4,
    }
    champion_file = checkpoint_dir / MILESTONE_SPECS[4][0]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in stage_keys:
                    loaded_label = try_activate_stage(
                        stage_keys[event.key], agent, checkpoint_dir
                    )
                    if loaded_label is not None:
                        stage_label = loaded_label
                        eval_mode = True
                        turbo_mode = False
                        if not reward_history:
                            reward_history = load_saved_reward_curve(checkpoint_dir)
                        state = env.reset()
                elif event.key == pygame.K_SPACE:
                    eval_mode = False
                    stage_label = ""
                    turbo_mode = not turbo_mode
                elif event.key == pygame.K_e:
                    eval_mode = not eval_mode
                    if eval_mode and agent.total_steps == 0 and champion_file.exists():
                        agent.load_checkpoint(champion_file)
                        stage_label = MILESTONE_SPECS[4][1]
                elif event.key == pygame.K_r:
                    show_rays = not show_rays
                elif event.key == pygame.K_s:
                    agent.save_checkpoint(champion_file)
                elif event.key == pygame.K_l and champion_file.exists():
                    agent.load_checkpoint(champion_file)
                    stage_label = MILESTONE_SPECS[4][1]

        steps_this_frame = 240 if (turbo_mode and not eval_mode) else 1
        t0 = time.perf_counter()
        for _ in range(steps_this_frame):
            last_action, last_q = agent.select_action(state, greedy=eval_mode)
            res = env.step(last_action, endless=eval_mode)
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
        mode = (
            stage_label
            if (eval_mode and stage_label)
            else ("GREEDY EVAL" if eval_mode else ("TURBO TRAIN" if turbo_mode else "LIVE 60 FPS"))
        )
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
    if not torch.cuda.is_available():
        torch.set_num_threads(1)

    arena_cfg = ArenaConfig()
    dqn_cfg = DQNConfig(obs_dim=arena_cfg.observation_dim, action_dim=arena_cfg.action_dim)
    env = RaycastArenaEnv(arena_cfg, seed=args.seed)
    agent = DQNAgent(dqn_cfg, seed=args.seed)

    ckpt_dir = Path(args.load).parent if args.load else DEFAULT_CHECKPOINT_DIR
    stage_label = ""
    start_eval = bool(args.eval)

    if args.stage in MILESTONE_SPECS:
        loaded = try_activate_stage(args.stage, agent, ckpt_dir)
        if loaded is not None:
            stage_label = loaded
            start_eval = True
    elif args.load and Path(args.load).exists():
        agent.load_checkpoint(args.load)
        stage_label = "CUSTOM CHECKPOINT"
    elif args.eval:
        loaded = try_activate_stage(4, agent, ckpt_dir)
        if loaded is not None:
            stage_label = loaded

    if args.headless:
        run_training_session(env, agent, episodes=args.episodes, checkpoint_dir=ckpt_dir)
    else:
        run_interactive_lab(
            env,
            agent,
            checkpoint_dir=ckpt_dir,
            start_eval=start_eval,
            initial_stage_label=stage_label,
        )


if __name__ == "__main__":
    main()
