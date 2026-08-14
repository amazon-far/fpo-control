# FPO++ Run Observatory

Scan FPO++ training log directories, grade runs against paper baselines, and flag reward cliffs.

No GPU or Isaac Sim required — works from `logs/isaaclab_fpo/*` tensorboard artifacts.

## Install

```bash
pip install -e tools/fpo_training_receipts
pip install -e tools/fpo_run_observatory
```

## Usage

```bash
# Scan all runs under a log root
fpo-observatory scan logs/isaaclab_fpo/g1_flat_flow --output-dir site/

# Open report
open site/index.html
```

## Output

- `index.html` — visual dashboard with green/yellow/red health grades
- `report.json` — machine-readable run receipts

## Grading

| Grade | Meaning |
|-------|---------|
| Green | No cliff; peak reward within 85% of paper target |
| Yellow | Insufficient data or below target |
| Red | Cliff detected (reward collapse after plateau) |

Baselines from `isaaclab_experiments/README.md` (4096 envs).
