#!/usr/bin/env bash
# Reproduce fpo-control#4 G1 locomotion reward cliff (Linux + NVIDIA GPU required).
set -euo pipefail

cd "$(dirname "$0")/.."
source source_env.sh

NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-42}"

echo "=== Baseline: 2000 iters (expected ~37 return, no cliff) ==="
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations 2000 \
    --seed "$SEED" \
    --run_name cliff_baseline_2k

echo "=== Repro: 5000 iters with default hyperparams (cliff expected ~4k) ==="
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations 5000 \
    --seed "$SEED" \
    --run_name cliff_repro_5k_default \
    agent.algorithm.ema_decay=0.95 \
    agent.algorithm.num_learning_epochs=32 \
    agent.algorithm.normalize_advantage=true

echo "=== Fix: 5000 iters with long-run stability overrides (auto-applied) ==="
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations 5000 \
    --seed "$SEED" \
    --run_name cliff_fix_5k_stable

echo "=== Doctor receipts ==="
for run in logs/isaaclab_fpo/g1_flat_flow/*cliff*; do
    python -m fpo_training_receipts doctor "$run" --write || true
done
