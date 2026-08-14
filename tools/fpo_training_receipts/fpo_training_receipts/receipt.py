"""Build training receipts from FPO++ log directories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from fpo_training_receipts.cliff import CliffReport, detect_reward_cliff


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _parse_tensorboard_rewards(log_dir: Path) -> list[float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []

    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return []

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    tag = "Train/mean_reward"
    if tag not in tags:
        return []

    events = acc.Scalars(tag)
    if not events:
        return []

    max_step = max(e.step for e in events)
    series = [0.0] * (max_step + 1)
    for e in events:
        if 0 <= e.step < len(series):
            series[e.step] = e.value
    return series


def build_receipt(log_dir: Path) -> dict[str, Any]:
    params_dir = log_dir / "params"
    agent = _load_yaml(params_dir / "agent.yaml")

    algorithm = agent.get("algorithm", {})
    policy = agent.get("policy", {})

    rewards = _parse_tensorboard_rewards(log_dir)
    cliff: CliffReport = detect_reward_cliff(rewards) if rewards else CliffReport(
        status="insufficient_data",
        peak_iteration=None,
        peak_reward=None,
        cliff_iteration=None,
        cliff_drop_fraction=None,
        message="No Train/mean_reward series found",
    )

    checkpoints = sorted(
        int(m.group(1))
        for p in log_dir.glob("model_*.pt")
        if (m := re.match(r"model_(\d+)\.pt", p.name))
    )

    return {
        "log_dir": str(log_dir),
        "experiment_name": agent.get("experiment_name"),
        "max_iterations": agent.get("max_iterations"),
        "num_envs": agent.get("num_envs"),
        "hyperparams": {
            "learning_rate": algorithm.get("learning_rate"),
            "num_learning_epochs": algorithm.get("num_learning_epochs"),
            "n_samples_per_action": algorithm.get("n_samples_per_action"),
            "ema_decay": algorithm.get("ema_decay"),
            "ema_warmup_steps": algorithm.get("ema_warmup_steps"),
            "normalize_advantage": algorithm.get("normalize_advantage"),
            "schedule": algorithm.get("schedule"),
            "clip_param": algorithm.get("clip_param"),
            "trust_region_mode": algorithm.get("trust_region_mode"),
            "sampling_steps": policy.get("sampling_steps"),
            "flow_eval_modes": agent.get("flow_eval_modes"),
        },
        "checkpoints": checkpoints,
        "reward_series_length": len(rewards),
        "cliff": {
            "status": cliff.status,
            "peak_iteration": cliff.peak_iteration,
            "peak_reward": cliff.peak_reward,
            "cliff_iteration": cliff.cliff_iteration,
            "cliff_drop_fraction": cliff.cliff_drop_fraction,
            "message": cliff.message,
        },
    }


def write_receipt(log_dir: Path, output: Path | None = None) -> Path:
    receipt = build_receipt(log_dir)
    out = output or (log_dir / "training_receipt.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    return out
