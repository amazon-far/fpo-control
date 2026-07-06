"""Detect reward cliff patterns in training curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CliffReport:
    status: str  # "healthy" | "cliff_detected" | "insufficient_data"
    peak_iteration: int | None
    peak_reward: float | None
    cliff_iteration: int | None
    cliff_drop_fraction: float | None
    message: str


def detect_reward_cliff(
    rewards: Sequence[float],
    *,
    min_peak_reward: float = 10.0,
    min_plateau_iters: int = 100,
    drop_threshold: float = 0.30,
    window: int = 50,
) -> CliffReport:
    """Detect a performance cliff after a reward plateau."""
    if len(rewards) < min_plateau_iters + window:
        return CliffReport(
            status="insufficient_data",
            peak_iteration=None,
            peak_reward=None,
            cliff_iteration=None,
            cliff_drop_fraction=None,
            message=f"Need at least {min_plateau_iters + window} iterations, got {len(rewards)}",
        )

    peak_reward = max(rewards)
    peak_iteration = rewards.index(peak_reward)

    if peak_reward < min_peak_reward:
        return CliffReport(
            status="insufficient_data",
            peak_iteration=peak_iteration,
            peak_reward=peak_reward,
            cliff_iteration=None,
            cliff_drop_fraction=None,
            message=f"Peak reward {peak_reward:.2f} below min_peak_reward={min_peak_reward}",
        )

    search_start = peak_iteration + min_plateau_iters
    for start in range(search_start, len(rewards) - window + 1):
        window_mean = sum(rewards[start : start + window]) / window
        drop = (peak_reward - window_mean) / max(peak_reward, 1e-8)
        if drop >= drop_threshold:
            return CliffReport(
                status="cliff_detected",
                peak_iteration=peak_iteration,
                peak_reward=peak_reward,
                cliff_iteration=start,
                cliff_drop_fraction=drop,
                message=(
                    f"Reward dropped {drop * 100:.1f}% from peak {peak_reward:.2f} "
                    f"(iter {peak_iteration}) to window mean {window_mean:.2f} (iter {start})"
                ),
            )

    return CliffReport(
        status="healthy",
        peak_iteration=peak_iteration,
        peak_reward=peak_reward,
        cliff_iteration=None,
        cliff_drop_fraction=None,
        message=f"No cliff detected; peak reward {peak_reward:.2f} at iter {peak_iteration}",
    )
