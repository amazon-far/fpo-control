"""Generate self-contained HTML observatory reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _grade_run(receipt: dict[str, Any]) -> tuple[str, str]:
    cliff = receipt.get("cliff", {})
    status = cliff.get("status", "insufficient_data")
    peak = cliff.get("peak_reward")
    baseline = receipt.get("baseline")

    if status == "cliff_detected":
        return "red", "Cliff detected — reward collapsed after plateau"
    if status == "insufficient_data":
        return "yellow", cliff.get("message", "Insufficient log data")
    if baseline and peak is not None:
        target = baseline.get("target_return", 0)
        if peak >= target * 0.85:
            return "green", f"Healthy — peak {peak:.1f} near target {target:.1f}"
        return "yellow", f"Below target — peak {peak:.1f} vs {target:.1f}"
    return "green", cliff.get("message", "No cliff detected")


def render_report(runs: list[dict[str, Any]], title: str = "FPO++ Run Observatory") -> str:
    cards: list[str] = []
    for run in runs:
        grade, reason = _grade_run(run)
        name = html.escape(Path(run["log_dir"]).name)
        cliff = run.get("cliff", {})
        hp = run.get("hyperparams", {})
        baseline = run.get("baseline")
        cards.append(
            f"""
            <article class="card {grade}">
              <header>
                <h2>{name}</h2>
                <span class="badge {grade}">{grade.upper()}</span>
              </header>
              <p class="reason">{html.escape(reason)}</p>
              <dl>
                <dt>Experiment</dt><dd>{html.escape(str(run.get('experiment_name', '—')))}</dd>
                <dt>Max iters</dt><dd>{run.get('max_iterations', '—')}</dd>
                <dt>Peak reward</dt><dd>{cliff.get('peak_reward', '—')}</dd>
                <dt>Peak iter</dt><dd>{cliff.get('peak_iteration', '—')}</dd>
                <dt>EMA decay</dt><dd>{hp.get('ema_decay', '—')}</dd>
                <dt>Epochs</dt><dd>{hp.get('num_learning_epochs', '—')}</dd>
                <dt>Checkpoints</dt><dd>{len(run.get('checkpoints', []))}</dd>
              </dl>
              {f"<p class='baseline'>Paper target: {baseline['target_return']} @ {baseline['max_iterations']} iters</p>" if baseline else ""}
              <footer><code>{html.escape(run['log_dir'])}</code></footer>
            </article>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1117; --card: #1a1d27; --text: #e8eaed; --muted: #9aa0a6;
      --green: #34d399; --yellow: #fbbf24; --red: #f87171; --accent: #60a5fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; }}
    h1 {{ margin: 0 0 0.25rem; font-size: 1.75rem; }}
    .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
    .grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }}
    .card {{ background: var(--card); border-radius: 12px; padding: 1.25rem; border: 1px solid #2a2f3a; }}
    .card.green {{ border-left: 4px solid var(--green); }}
    .card.yellow {{ border-left: 4px solid var(--yellow); }}
    .card.red {{ border-left: 4px solid var(--red); }}
    .card header {{ display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }}
    .card h2 {{ margin: 0; font-size: 1rem; word-break: break-all; }}
    .badge {{ font-size: 0.7rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 999px; }}
    .badge.green {{ background: #064e3b; color: var(--green); }}
    .badge.yellow {{ background: #78350f; color: var(--yellow); }}
    .badge.red {{ background: #7f1d1d; color: var(--red); }}
    .reason {{ color: var(--muted); font-size: 0.9rem; }}
    dl {{ display: grid; grid-template-columns: auto 1fr; gap: 0.25rem 1rem; font-size: 0.85rem; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; }}
    .baseline {{ color: var(--accent); font-size: 0.85rem; }}
    footer code {{ font-size: 0.7rem; color: var(--muted); word-break: break-all; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="subtitle">{len(runs)} run(s) scanned — grades runs vs paper baselines and cliff patterns</p>
  <div class="grid">{''.join(cards) if cards else '<p>No runs found.</p>'}</div>
</body>
</html>"""


def write_report(runs: list[dict[str, Any]], output: Path, title: str = "FPO++ Run Observatory") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(runs, title=title))
    return output


def write_json_report(runs: list[dict[str, Any]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    return output
