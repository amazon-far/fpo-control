# FPO++ IsaacLab Experiments

FPO++ replaces Gaussian action distributions in PPO with a learned flow model
that maps noise to actions via an ODE. This repository integrates FPO++ as an
[NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) extension for
velocity-conditioned locomotion (6+ robots).

## Setup

**Prerequisites**: Linux, NVIDIA GPU with CUDA 12.1+.

```bash
# Install everything (conda env, IsaacSim 4.5, IsaacLab, isaaclab_fpo).
bash setup_env.sh

# Activate the environment (run this at the start of every session).
source source_env.sh
```

This creates a conda environment named `isaaclab_fpo`.

## Training

All FPO++ training is launched via `isaaclab_fpo/scripts/train.py`. Per-task hyperparameters are defined in `isaaclab_fpo/isaaclab_fpo/task_cfgs.py`.

### Velocity-Conditioned Locomotion

```bash
# Unitree Go2.
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless

# Boston Dynamics Spot.
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-Spot-v0 --headless

# Unitree H1 (humanoid).
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-H1-v0 --headless

# Unitree G1 (humanoid).
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
```

**Expected training curves** (4096 envs):

![Training curves](expected_training_curves_locomotion.png)

| Robot | Iterations | Final Return |
|-------|-----------|--------------|
| Go2   | 1500      | ~40          |
| Spot  | 1500      | ~315         |
| H1    | 2000      | ~38          |
| G1    | 2000      | ~37          |

### Extended G1 training (>2000 iterations)

Issue [#4](https://github.com/amazon-far/fpo-control/issues/4) reports a reward cliff near iteration ~4000 when training to 5000 with default hyperparams. G1/H1 configs are tuned for **≤2000 iterations** (`ema_decay=0.95`, 32 learning epochs, advantage normalization).

**Recommended:** stop at 2000 iterations for paper-matching curves.

**If you need longer runs**, `train.py` auto-applies stability overrides when `--max_iterations > 2000`:

- `ema_decay`: 0.95 → 0.99
- `num_learning_epochs`: 32 → 16
- `normalize_advantage`: true → false

Or set manually:

```bash
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless \
    --num_envs 4096 --max_iterations 5000 \
    agent.algorithm.ema_decay=0.99 \
    agent.algorithm.num_learning_epochs=16 \
    agent.algorithm.normalize_advantage=false
```

After training, check for cliffs with **FPO++ Training Receipts**:

```bash
pip install -e tools/fpo_training_receipts
pip install -e tools/fpo_run_observatory
python -m fpo_training_receipts doctor logs/isaaclab_fpo/g1_flat_flow/<run_dir> --write
fpo-observatory scan logs/isaaclab_fpo/g1_flat_flow --output-dir observatory_out
```

See `agent/FPO_CLIFF_RC.md` for reproduction, profiling checklist, and root-cause analysis.

**Evaluation returns** (checkpoints evaluated with zero and random initial noise):

![Eval curves](expected_eval_curves_locomotion.png)

### Whole-Body Motion Tracking (G1)

```bash
# Train on a specific motion (default: walk1_subject1).
python isaaclab_fpo/scripts/train.py --task Tracking-Flat-G1-v0 --headless

# Train on a different motion.
python isaaclab_fpo/scripts/train.py --task Tracking-Flat-G1-v0 --headless \
    env.commands.motion.motion_file=whole_body_tracking_reference_data/dance1_subject2.npz
```

Available motions (in `whole_body_tracking_reference_data/`):
`walk1_subject1`, `run1_subject2`, `jumps1_subject1`, `dance1_subject1`,
`dance1_subject2`, `fight1_subject2`, `fallAndGetUp1_subject1`.

**Expected training curves** (4096 envs):

![Tracking training curves](expected_training_curves_tracking.png)

### Common Options

```bash
# Override num envs, max iterations, seed.
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
    --num_envs 4096 --max_iterations 2000 --seed 42

# Log to Weights & Biases.
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
    --logger wandb --log_project_name my-project --run_name trial_01

# Override hyperparameters via positional args (useful for sweeps).
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
    agent.algorithm.learning_rate=3e-4 \
    agent.algorithm.n_samples_per_action=32
```

| Flag                 | Description                                      |
| -------------------- | ------------------------------------------------ |
| `--logger wandb`     | Enable W&B logging (default: tensorboard)        |
| `--log_project_name` | W&B project name (default: `isaaclab`)           |
| `--run_name`         | Suffix appended to the timestamped run directory |
| `--num_envs`         | Number of parallel environments                  |
| `--max_iterations`   | Override max training iterations                 |
| `--seed`             | Random seed (`-1` for random)                    |

## Playback (Viser)

Visualize a trained policy in the browser using [Viser](https://viser.studio/):

```bash
# Load checkpoint from a local path.
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --checkpoint logs/isaaclab_fpo/unitree_go2_flat_flow/2025-01-01_00-00-00/model_1500.pt \
    --headless --viser --num_envs 1

# Load checkpoint from W&B (entity/project/run_id from the run's URL).
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --wandb-run-path my-entity/my-project/abc123xy \
    --wandb-checkpoint model_1500.pt \
    --headless --viser --num_envs 4

```

Then open `http://localhost:8080` in your browser.

The `--wandb-run-path` is the `entity/project/run_id` from your W&B run URL (e.g. `https://wandb.ai/my-entity/my-project/runs/abc123xy` → `my-entity/my-project/abc123xy`).

## Directory Layout

```
isaaclab_experiments/
├── isaaclab_fpo/                         # FPO++ package (algorithm + IsaacLab integration)
│   ├── scripts/
│   │   ├── train.py                     # Main training entry point
│   │   ├── play_with_viser.py           # Viser-based policy playback (browser)
│   │   ├── play.py                      # IsaacSim viewer playback
│   │   ├── play_plot.py                 # Playback with live reward plotting
│   ├── viser_assets/                    # Pre-extracted robot meshes for Viser
│   └── isaaclab_fpo/
│       ├── task_cfgs.py                 # Per-task FPO++ hyperparameters (TASK_CONFIGS registry)
│       ├── rl_cfg.py                    # FPO++ config dataclasses
│       ├── algorithms/fpo.py            # FPO++ algorithm (flow-based PPO)
│       ├── modules/actor_critic.py      # Flow actor + value critic networks
│       ├── runners/on_policy_runner.py  # Training loop with EMA, eval, multi-GPU
│       ├── storage/rollout_storage.py   # Rollout buffer with CFM loss storage
│       ├── wrapper.py                   # VecEnv wrapper for IsaacLab
│       ├── cli_args.py                  # CLI argument helpers
│       └── patches.py                   # IsaacLab monkey-patches for sweep support
│
├── thirdparty/
│   ├── IsaacLab/                        # NVIDIA Isaac Lab (git submodule)
│   │   └── source/
│   │       ├── isaaclab/                # Core framework
│   │       ├── isaaclab_tasks/          # Task definitions (locomotion, etc.)
│   │       └── isaaclab_assets/         # Robot USD assets & configs
│   └── whole_body_tracking/             # Motion tracking envs (git submodule)
│
├── whole_body_tracking_reference_data/   # LAFAN1 motion NPZ files (generated by setup)
├── expected_training_curves_locomotion.png
├── expected_eval_curves_locomotion.png
├── expected_training_curves_tracking.png
├── expected_eval_curves_tracking.png
├── setup_env.sh                          # One-time environment setup
└── source_env.sh                         # Activate conda env
```

## Acknowledgements

The `isaaclab_fpo` package combines and adapts code from the following projects:

| Source | License | What we adapted |
|--------|---------|-----------------|
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (ETH Zurich + NVIDIA) | BSD-3-Clause | Actor-critic, on-policy runner, rollout storage, normalizer, logging utilities |
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) (NVIDIA) | BSD-3-Clause | VecEnv wrapper, config dataclasses, training/play/evaluate scripts, ONNX exporter |

`IsaacLab/` is included as a git submodule under its original license. The `isaaclab_fpo` package adapts code from rsl_rl and IsaacLab; original copyright headers are retained in all adapted files.
