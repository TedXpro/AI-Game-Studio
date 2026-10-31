"""Pygame-CE geometric arena renderer and live Double-DQN telemetry HUD."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pygame

from src.sim.entities import ArenaConfig, HitType
from src.sim.arena_env import ArenaSnapshot


ACTION_LABELS: tuple[str, ...] = ("COAST", "THRUST", "LEFT", "RIGHT", "THR+L", "THR+R")
PANEL_WIDTH: int = 320


@dataclass(frozen=True)
class TelemetryFrame:
    """Live training and policy metrics displayed on the right-hand HUD panel."""

    episode: int
    epsilon: float
    avg_reward: float
    best_reward: float
    steps_per_sec: float
    q_values: NDArray[np.float32]
    chosen_action: int
    reward_history: list[float]
    mode_label: str
    show_rays: bool = True


class ArenaVisualizer:
    """Renders 2D LiDAR geometry and neural Q-value telemetry without menu bloat."""

    def __init__(self, config: ArenaConfig, offscreen: bool = False) -> None:
        """Initializes Pygame fonts and either a display window or offscreen canvas.

        Args:
            config: Arena physical dimensions and entity radii.
            offscreen: When True, renders onto a memory surface without opening a window.
        """
        self.config = config
        self.offscreen = offscreen
        self.width = int(config.width) + PANEL_WIDTH
        self.height = int(config.height)

        pygame.init()
        pygame.font.init()
        if offscreen:
            self.surface = pygame.Surface((self.width, self.height))
        else:
            pygame.display.set_caption("AI Game Studio | 01 - Raycast Survival Arena (Double DQN)")
            self.surface = pygame.display.set_mode((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("consolas", 17, bold=True)
        self.font_body = pygame.font.SysFont("consolas", 14)
        self.font_small = pygame.font.SysFont("consolas", 12)

    def render(
        self,
        snapshot: ArenaSnapshot,
        telemetry: TelemetryFrame,
        fps_limit: int = 60,
    ) -> None:
        """Draws a complete arena and neural HUD frame and flips the display if active.

        Args:
            snapshot: Immutable geometric state of the simulation.
            telemetry: Current RL training metrics and Q-values.
            fps_limit: Target frame rate cap when running in interactive window mode.
        """
        self.surface.fill((12, 16, 24))
        self._draw_arena_grid()
        if telemetry.show_rays:
            self._draw_lidar_rays(snapshot)
        self._draw_entities(snapshot)
        self._draw_hud_panel(snapshot, telemetry)

        if not self.offscreen:
            pygame.display.flip()
            if fps_limit > 0:
                self.clock.tick(fps_limit)

    def save_screenshot(self, path: str | Path) -> None:
        """Writes the current rendered surface to a PNG image file.

        Args:
            path: Output image path.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.surface, str(dest))

    def close(self) -> None:
        """Cleanly shuts down the Pygame subsystem."""
        pygame.quit()

    def _draw_arena_grid(self) -> None:
        """Draws background spatial grid lines and the arena boundary frame."""
        w, h = int(self.config.width), int(self.config.height)
        for x in range(0, w, 80):
            pygame.draw.line(self.surface, (20, 27, 38), (x, 0), (x, h), 1)
        for y in range(0, h, 80):
            pygame.draw.line(self.surface, (20, 27, 38), (0, y), (w, y), 1)
        pygame.draw.rect(self.surface, (64, 92, 130), pygame.Rect(0, 0, w, h), 2)

    def _draw_lidar_rays(self, snapshot: ArenaSnapshot) -> None:
        """Renders color-coded LiDAR rays and surface contact markers."""
        origin = (int(snapshot.agent.position[0]), int(snapshot.agent.position[1]))
        palette = {
            int(HitType.NONE): (44, 56, 74),
            int(HitType.WALL): (215, 160, 65),
            int(HitType.HAZARD): (245, 75, 90),
            int(HitType.CORE): (55, 235, 195),
        }
        for idx in range(snapshot.rays.endpoints.shape[0]):
            hit = int(snapshot.rays.hit_types[idx])
            end = (int(snapshot.rays.endpoints[idx, 0]), int(snapshot.rays.endpoints[idx, 1]))
            color = palette.get(hit, (44, 56, 74))
            pygame.draw.line(self.surface, color, origin, end, 1)
            if hit != int(HitType.NONE):
                pygame.draw.circle(self.surface, color, end, 3)

    def _draw_entities(self, snapshot: ArenaSnapshot) -> None:
        """Draws energy cores, hazard mines, and the directional scout drone."""
        core_r = int(self.config.core_radius)
        for core in snapshot.cores:
            pos = (int(core[0]), int(core[1]))
            pygame.draw.circle(self.surface, (22, 95, 82), pos, core_r + 5)
            pygame.draw.circle(self.surface, (60, 240, 190), pos, core_r)
            pygame.draw.circle(self.surface, (210, 255, 245), pos, max(3, core_r // 3))

        haz_r = int(self.config.hazard_radius)
        for hazard in snapshot.hazards:
            pos = (int(hazard[0]), int(hazard[1]))
            pygame.draw.circle(self.surface, (95, 26, 38), pos, haz_r + 4)
            pygame.draw.circle(self.surface, (235, 65, 82), pos, haz_r)
            pygame.draw.circle(self.surface, (255, 165, 175), pos, max(3, haz_r // 3), 1)

        ax, ay = float(snapshot.agent.position[0]), float(snapshot.agent.position[1])
        ar = float(self.config.agent_radius)
        h = snapshot.agent.heading
        nose = (ax + math.cos(h) * (ar + 4.0), ay + math.sin(h) * (ar + 4.0))
        left = (ax + math.cos(h + 2.45) * ar, ay + math.sin(h + 2.45) * ar)
        right = (ax + math.cos(h - 2.45) * ar, ay + math.sin(h - 2.45) * ar)
        pygame.draw.circle(self.surface, (32, 68, 115), (int(ax), int(ay)), int(ar), 2)
        pygame.draw.polygon(self.surface, (95, 185, 255), [nose, left, right])

    def _draw_hud_panel(self, snapshot: ArenaSnapshot, telemetry: TelemetryFrame) -> None:
        """Renders the telemetry sidebar containing metrics, Q-bars, and reward curve."""
        x0 = int(self.config.width)
        pygame.draw.rect(self.surface, (17, 22, 33), pygame.Rect(x0, 0, PANEL_WIDTH, self.height))
        pygame.draw.line(self.surface, (55, 75, 105), (x0, 0), (x0, self.height), 2)

        pad = x0 + 16
        self.surface.blit(self.font_title.render("NEURAL TELEMETRY", True, (225, 235, 250)), (pad, 14))
        badge_color = (60, 235, 185) if "EVAL" in telemetry.mode_label else (255, 185, 70)
        self.surface.blit(self.font_body.render(f"MODE: {telemetry.mode_label}", True, badge_color), (pad, 38))

        stats = [
            f"Episode      : {telemetry.episode:5d}",
            f"Epsilon      : {telemetry.epsilon:5.3f}",
            f"Ep Cores     : {snapshot.episode_cores:5d}",
            f"Ep Reward    : {snapshot.episode_reward:7.2f}",
            f"Avg Rwd (50) : {telemetry.avg_reward:7.2f}",
            f"Best Reward  : {telemetry.best_reward:7.2f}",
            f"Sim Speed    : {telemetry.steps_per_sec:6.0f} st/s",
        ]
        for idx, line in enumerate(stats):
            self.surface.blit(self.font_body.render(line, True, (180, 196, 218)), (pad, 68 + idx * 20))

        self._draw_q_bars(pad, 222, telemetry.q_values, telemetry.chosen_action)
        self._draw_sparkline(pad, 412, PANEL_WIDTH - 32, 115, telemetry.reward_history)

        controls = "[1-4] Stages  [SPACE] Turbo  [E] Eval  [R] Rays"
        self.surface.blit(self.font_small.render(controls, True, (125, 145, 175)), (pad, self.height - 26))

    def _draw_q_bars(
        self,
        x: int,
        y: int,
        q_values: NDArray[np.float32],
        chosen_action: int,
    ) -> None:
        """Draws real-time relative Q(s, a) advantage bars for all six actions."""
        self.surface.blit(self.font_body.render("LIVE Q(s, a) ADVANTAGE", True, (220, 232, 248)), (x, y))
        centered = q_values - float(np.mean(q_values))
        scale = max(1.0, float(np.max(np.abs(centered))))
        bar_mid = x + 150
        max_half = 82

        for idx, label in enumerate(ACTION_LABELS):
            row_y = y + 24 + idx * 25
            active = idx == chosen_action
            txt_color = (70, 245, 195) if active else (150, 168, 190)
            self.surface.blit(self.font_small.render(label, True, txt_color), (x, row_y + 2))
            pygame.draw.rect(self.surface, (28, 36, 52), pygame.Rect(bar_mid - max_half, row_y, max_half * 2, 16))
            span = int((float(centered[idx]) / scale) * max_half)
            bar_color = (60, 235, 185) if active else (75, 130, 215)
            rect_x = bar_mid if span >= 0 else bar_mid + span
            pygame.draw.rect(self.surface, bar_color, pygame.Rect(rect_x, row_y + 2, max(2, abs(span)), 12))
            val_txt = self.font_small.render(f"{float(q_values[idx]):+5.1f}", True, txt_color)
            self.surface.blit(val_txt, (bar_mid + max_half + 6, row_y + 2))

    def _draw_sparkline(self, x: int, y: int, w: int, h: int, history: list[float]) -> None:
        """Draws a rolling episode reward curve inside the telemetry panel."""
        self.surface.blit(self.font_body.render("EPISODE REWARD TREND", True, (220, 232, 248)), (x, y))
        box = pygame.Rect(x, y + 22, w, h)
        pygame.draw.rect(self.surface, (24, 31, 46), box)
        pygame.draw.rect(self.surface, (46, 62, 88), box, 1)
        recent = history[-80:]
        if len(recent) < 2:
            return
        lo, hi = min(recent), max(recent)
        span = max(1.0, hi - lo)
        pts = [
            (
                x + int((i / (len(recent) - 1)) * (w - 8)) + 4,
                y + 22 + h - 6 - int(((val - lo) / span) * (h - 12)),
            )
            for i, val in enumerate(recent)
        ]
        pygame.draw.lines(self.surface, (85, 195, 255), False, pts, 2)
