# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with FPO."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

import viser # HACK: needs to happen *before* Isaac stuff to prevent websockets package error. Super annoying.
# The reason is that Isaac does some crazy path stuff, which will force an older version of websockets to be imported.

from isaaclab.app import AppLauncher

# local imports
from isaaclab_fpo import cli_args  # isort: skip

# Hack: wandb sweep doesn't work with "--seed" as a key...
orig_argv = sys.argv.copy()
sys.argv = [arg if not arg.startswith("seed=") else "--" + arg for arg in sys.argv]

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with FPO.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
# append FPO cli arguments
cli_args.add_fpo_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab_fpo.patches import apply_isaaclab_patches
apply_isaaclab_patches()

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab_fpo.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401 — registers motion tracking envs
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_fpo.task_cfgs import TASK_CONFIGS

# Humanoid locomotion tasks tuned for <=2000 iters; need stability overrides beyond that.
HUMANOID_LOCOMOTION_TASKS = frozenset({
    "Isaac-Velocity-Flat-G1-v0",
    "Isaac-Velocity-Flat-G1-Play-v0",
    "Isaac-Velocity-Flat-H1-v0",
    "Isaac-Velocity-Flat-H1-Play-v0",
})

LONG_RUN_ITERATION_THRESHOLD = 2000


def apply_long_run_stability_overrides(
    agent_cfg, task: str, max_iterations: int, *, algorithm_overridden: bool
) -> None:
    """Adjust algorithm hyperparams when training humanoids past the tuned budget.

    See agent/FPO_CLIFF_RC.md and fpo-control#4. Skipped when the user explicitly
    set agent.algorithm.* overrides (e.g. to reproduce the cliff).
    """
    if task not in HUMANOID_LOCOMOTION_TASKS:
        return
    if max_iterations <= LONG_RUN_ITERATION_THRESHOLD:
        return
    if algorithm_overridden:
        print(
            f"[INFO] max_iterations={max_iterations} > {LONG_RUN_ITERATION_THRESHOLD}: "
            "skipping auto stability overrides (explicit agent.algorithm.* overrides detected)"
        )
        return

    algo = agent_cfg.algorithm
    changes: list[str] = []

    if algo.ema_decay == 0.95:
        algo.ema_decay = 0.99
        changes.append("ema_decay 0.95 -> 0.99")
    if algo.num_learning_epochs == 32:
        algo.num_learning_epochs = 16
        changes.append("num_learning_epochs 32 -> 16")
    if algo.normalize_advantage is True:
        algo.normalize_advantage = False
        changes.append("normalize_advantage True -> False")

    if changes:
        print(
            f"[INFO] max_iterations={max_iterations} > {LONG_RUN_ITERATION_THRESHOLD} "
            f"on {task}: applying long-run stability overrides: {', '.join(changes)}"
        )
        print(
            "[INFO] For paper-matching G1 curves, use --max_iterations 2000 instead. "
            "See isaaclab_experiments/README.md#extended-g1-training."
        )


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _parse_sweep_overrides():
    """Parse W&B sweep overrides from sys.argv positional args.

    W&B sweeps pass parameters as positional args like:
        agent.policy.actor_hidden_dims=[256,256,256]  agent.algorithm.clip_param=0.05

    Returns two dicts: one for 'agent.*' overrides and one for 'env.*' overrides,
    suitable for passing to configclass.from_dict().
    """
    import ast

    agent_overrides = {}
    env_overrides = {}

    for arg in sys.argv[1:]:
        if "=" not in arg or arg.startswith("-"):
            continue

        key, value_str = arg.split("=", 1)

        # Parse the value
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Handle booleans and strings that ast.literal_eval can't parse
            if value_str.lower() == "true":
                value = True
            elif value_str.lower() == "false":
                value = False
            else:
                value = value_str

        # Route to agent or env overrides
        if key.startswith("agent."):
            parts = key[len("agent."):].split(".")
        elif key.startswith("env."):
            parts = key[len("env."):].split(".")
        else:
            continue

        # Build nested dict from dotted path
        d = agent_overrides if key.startswith("agent.") else env_overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    return agent_overrides, env_overrides


def main():
    """Train with FPO agent."""
    # parse env and agent configs from registries
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_fpo_cfg(args_cli.task, args_cli)

    # Apply W&B sweep overrides (positional args like agent.policy.actor_hidden_dims=[256,256,256])
    agent_overrides, env_overrides = _parse_sweep_overrides()
    if agent_overrides:
        agent_cfg.from_dict(agent_overrides)
    if env_overrides:
        env_cfg.from_dict(env_overrides)
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )
    apply_long_run_stability_overrides(
        agent_cfg,
        args_cli.task,
        agent_cfg.max_iterations,
        algorithm_overridden=bool(agent_overrides.get("algorithm")),
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "isaaclab_fpo", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if agent_cfg.load_run != '.*':
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        else:
            resume_path = agent_cfg.load_checkpoint

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for FPO
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from FPO
    sys.argv = orig_argv  # restore original sys.argv. This will ensure that the wandb run records the correct arguments.
    runner = OnPolicyRunner(
        env, agent_cfg, log_dir=log_dir, device=agent_cfg.device
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
