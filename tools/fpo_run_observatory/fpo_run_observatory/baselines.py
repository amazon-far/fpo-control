"""Paper/reference baselines for locomotion tasks (from isaaclab_experiments README)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskBaseline:
    task_id: str
    experiment_name: str
    max_iterations: int
    target_return: float
    num_envs: int = 4096


LOCOMOTION_BASELINES: dict[str, TaskBaseline] = {
    "Isaac-Velocity-Flat-Unitree-Go2-v0": TaskBaseline(
        task_id="Isaac-Velocity-Flat-Unitree-Go2-v0",
        experiment_name="unitree_go2_flat_flow",
        max_iterations=1500,
        target_return=40.0,
    ),
    "Isaac-Velocity-Flat-Spot-v0": TaskBaseline(
        task_id="Isaac-Velocity-Flat-Spot-v0",
        experiment_name="spot_flat_flow",
        max_iterations=1500,
        target_return=315.0,
    ),
    "Isaac-Velocity-Flat-H1-v0": TaskBaseline(
        task_id="Isaac-Velocity-Flat-H1-v0",
        experiment_name="h1_flat_flow",
        max_iterations=2000,
        target_return=38.0,
    ),
    "Isaac-Velocity-Flat-G1-v0": TaskBaseline(
        task_id="Isaac-Velocity-Flat-G1-v0",
        experiment_name="g1_flat_flow",
        max_iterations=2000,
        target_return=37.0,
    ),
}


def baseline_for_experiment(experiment_name: str | None) -> TaskBaseline | None:
    if not experiment_name:
        return None
    for baseline in LOCOMOTION_BASELINES.values():
        if baseline.experiment_name == experiment_name:
            return baseline
    return None
