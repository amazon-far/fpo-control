"""Scan log directories and build observatory run records."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fpo_run_observatory.baselines import baseline_for_experiment
from fpo_training_receipts.receipt import build_receipt


def find_run_dirs(logs_root: Path) -> list[Path]:
    """Find directories that look like FPO training runs."""
    logs_root = logs_root.expanduser().resolve()
    if (logs_root / "params" / "agent.yaml").exists():
        return [logs_root]

    runs: list[Path] = []
    for agent_yaml in logs_root.rglob("params/agent.yaml"):
        runs.append(agent_yaml.parent.parent)
    return sorted(set(runs), key=lambda p: p.stat().st_mtime, reverse=True)


def scan_run(log_dir: Path) -> dict[str, Any]:
    receipt = build_receipt(log_dir)
    baseline = baseline_for_experiment(receipt.get("experiment_name"))
    if baseline:
        receipt["baseline"] = asdict(baseline)
    grade_status, grade_reason = _grade(receipt, baseline)
    receipt["grade"] = {"status": grade_status, "reason": grade_reason}
    return receipt


def _grade(receipt: dict[str, Any], baseline) -> tuple[str, str]:
    cliff_status = receipt.get("cliff", {}).get("status")
    peak = receipt.get("cliff", {}).get("peak_reward")

    if cliff_status == "cliff_detected":
        return "red", "Cliff detected"
    if cliff_status == "insufficient_data":
        return "yellow", receipt["cliff"].get("message", "Insufficient data")
    if baseline and peak is not None:
        if peak >= baseline.target_return * 0.85:
            return "green", f"Near paper target ({peak:.1f} / {baseline.target_return})"
        return "yellow", f"Below paper target ({peak:.1f} / {baseline.target_return})"
    return "green", "No cliff detected"


def scan_logs(logs_root: Path) -> list[dict[str, Any]]:
    return [scan_run(d) for d in find_run_dirs(logs_root)]
