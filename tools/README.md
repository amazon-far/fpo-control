# FPO++ Training Tools

GPU-free observability for [amazon-far/fpo-control](https://github.com/amazon-far/fpo-control) training runs.

## Packages

| Package | CLI | Purpose |
|---------|-----|---------|
| [`fpo_training_receipts`](fpo_training_receipts/) | `python -m fpo_training_receipts doctor` | Per-run cliff detection + hyperparam receipt |
| [`fpo_run_observatory`](fpo_run_observatory/) | `fpo-observatory scan` | Multi-run HTML dashboard vs paper baselines |

## Quick start

```bash
pip install -e tools/fpo_training_receipts
pip install -e tools/fpo_run_observatory

# Single run
python -m fpo_training_receipts doctor logs/isaaclab_fpo/g1_flat_flow/<run_dir> --write

# All runs under an experiment
fpo-observatory scan logs/isaaclab_fpo/g1_flat_flow --output-dir observatory_out
open observatory_out/index.html
```

No Isaac Sim or GPU required — reads tensorboard events and `params/agent.yaml` from log dirs.
