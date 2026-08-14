# G1 Locomotion Reward Cliff — Root Cause Analysis

**Issue:** [amazon-far/fpo-control#4](https://github.com/amazon-far/fpo-control/issues/4)  
**Symptom:** Mean episode reward rises to ~35–37 by iter 1500–2000, then collapses near iter ~4000 when training is extended to 5000.  
**Repro:** `Isaac-Velocity-Flat-G1-v0`, 4096 envs, default hyperparams, `--max_iterations 5000`, single NVIDIA 3090.

## Reproduction command

```bash
cd isaaclab_experiments
source source_env.sh
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 5000 \
    --seed 42 \
    --logger wandb \
    --run_name cliff_repro
```

Baseline (no cliff, matches paper curves):

```bash
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 2000 \
    --seed 42
```

## Profiling checklist

Log these metrics (already emitted by FPO when using tensorboard/wandb):

| Metric | Path | What to look for at cliff |
|--------|------|---------------------------|
| Mean reward | `Train/mean_reward` | Sudden drop after plateau |
| Surrogate loss | `Loss/surrogate_loss` | Spike |
| Value loss | `Loss/value_loss` | Divergence |
| Learning rate | `Loss/learning_rate` | Flat at 1e-4 (fixed schedule) |
| Obs norm std | `Metrics/obs_norm_max_std` | Drift / collapse |
| Grad norm | `Metrics/mean_grad_norm_before_clip` | Spike |
| KL | `Metrics/kl` | Only if `schedule=adaptive` |

Use the receipt doctor:

```bash
python -m fpo_training_receipts doctor logs/isaaclab_fpo/g1_flat_flow/<run_dir>
```

## Root cause (code-level)

The cliff is **overtraining beyond the tuned iteration budget**, not an env-reset bug.

### 1. Hyperparams tuned for ≤2000 iterations

`G1FlatFlowPPORunnerCfg` sets `max_iterations=2000`. Algorithm defaults in `rl_cfg.py` document:

- `ema_decay=0.95` — "suitable for short training runs (~1500–2000 steps)"
- `num_learning_epochs=32` — humanoid override; 32 PPO epochs per rollout
- `schedule="fixed"` — LR stays at 1e-4 regardless of policy drift
- `normalize_advantage=True` — amplifies tiny return differences once the policy is near-optimal

The README expected curve plateaus at ~37 return by iter 2000. Continuing to 5000 applies **32 additional epochs × 3000 extra iterations** of updates on a policy that has already converged.

### 2. EMA / checkpoint mismatch

Checkpoints saved after EMA warmup (iter 500+) bake EMA actor weights (`on_policy_runner.py:save`). With `ema_decay=0.95` (effective window ~20 updates), late-stage weight churn causes growing divergence between live training weights and EMA shadow weights. Inference uses EMA-baked weights; training reward uses live weights — both can degrade after extended training.

### 3. Advantage normalization at convergence

`rollout_storage.py:compute_returns` normalizes advantages globally. When all envs perform similarly (high reward, low variance), normalized advantages become noisy and drive large CFM ratio updates (`fpo.py` surrogate with 32 samples per action).

### 4. Not the cause

- **4096 env count** — matches README; unrelated to cliff timing
- **KL adaptive schedule** — disabled by default (`schedule=fixed`)
- **Env reset bug** — no step-dependent curriculum found in G1 velocity task configs

## Fix / workaround

### Recommended (matches paper curves)

Stop at **2000 iterations** for G1 locomotion.

### Extended training (>2000 iters)

`train.py` auto-applies stability overrides when `--max_iterations > 2000` on humanoid locomotion tasks (unless manually overridden):

| Parameter | Default | Long-run |
|-----------|---------|----------|
| `ema_decay` | 0.95 | 0.99 |
| `num_learning_epochs` | 32 | 16 |
| `normalize_advantage` | True | False |

Manual override:

```bash
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless \
    --num_envs 4096 --max_iterations 5000 \
    agent.algorithm.ema_decay=0.99 \
    agent.algorithm.num_learning_epochs=16 \
    agent.algorithm.normalize_advantage=false
```

## Verification

1. Run repro command above on Linux + NVIDIA GPU
2. Compare `Train/mean_reward` at iter 2000, 4000, 5000
3. With fix: reward should not drop >30% from peak in any 500-iter window after iter 1500
4. Run `python -m fpo_training_receipts doctor <log_dir>` — should report `status: healthy`

## Hardware note

Issue reporter used RTX 3090, 4096 envs. Document your GPU in PR body; cliff is algorithmic, not GPU-specific.
