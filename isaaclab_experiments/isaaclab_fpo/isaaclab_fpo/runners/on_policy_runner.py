# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

from typing import TYPE_CHECKING

import isaaclab_fpo
from isaaclab_fpo.algorithms import FPO

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlOnPolicyRunnerCfg
from isaaclab_fpo.env import VecEnv
from isaaclab_fpo.modules import (
    ActorCritic,
    EmpiricalNormalization,
)
from isaaclab_fpo.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(
        self, env: VecEnv, train_cfg: FpoRslRlOnPolicyRunnerCfg, log_dir: str | None = None, device="cpu"
    ):
        self.cfg = train_cfg
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"
        else:
            self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[
                1
            ]
        else:
            num_privileged_obs = num_obs

        # initialize policy with config dataclass
        policy: ActorCritic = ActorCritic(
            num_obs, num_privileged_obs, self.env.num_actions, cfg=train_cfg.policy
        ).to(self.device)

        # initialize algorithm with config dataclass
        self.alg: FPO = FPO(
            policy, cfg=train_cfg.algorithm, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # store training configuration
        self.num_steps_per_env = train_cfg.num_steps_per_env
        self.save_interval = train_cfg.save_interval
        self.empirical_normalization = train_cfg.empirical_normalization
        self.randomize_reset_episode_progress = getattr(train_cfg, "randomize_reset_episode_progress", 0.0)

        # store post-training evaluation configuration
        self.enable_post_training_eval = getattr(self.cfg, "enable_post_training_eval", True)
        self.post_eval_checkpoint_interval = getattr(self.cfg, "post_eval_checkpoint_interval", 1)


        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1.0e8
            ).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[num_privileged_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(
                self.device
            )  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(
                self.device
            )  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [isaaclab_fpo.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = getattr(self.cfg, "logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from isaaclab_fpo.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.cfg.algorithm, self.cfg.policy
                )
            elif self.logger_type == "wandb":
                from isaaclab_fpo.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.cfg.algorithm, self.cfg.policy
                )
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError(
                    "Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'."
                )

            # Define custom WandB metrics for post-training eval
            if self.logger_type == "wandb" and self.enable_post_training_eval:
                import wandb
                wandb.define_metric("eval_iteration")
                # Define metrics for each eval mode
                eval_modes = getattr(self.cfg, "flow_eval_modes", ["zero", "fixed_seed", "random"])
                for mode in eval_modes:
                    wandb.define_metric(f"PostEval_{mode}/mean_reward", step_metric="eval_iteration")
                    wandb.define_metric(f"PostEval_{mode}/std_reward", step_metric="eval_iteration")
                    wandb.define_metric(f"PostEval_{mode}/mean_episode_length", step_metric="eval_iteration")
                    wandb.define_metric(f"PostEval_{mode}/std_episode_length", step_metric="eval_iteration")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        # Normalize before the first act() call (fixes #6).
        obs = self.obs_normalizer(obs)
        if self.privileged_obs_type is not None:
            privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        else:
            privileged_obs = obs
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_ereward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_ireward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.perf_counter()
            # Initialize timing accumulators
            env_step_time = 0.0
            action_time = 0.0
            process_time = 0.0
            
            # Rollout
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    action_start = time.perf_counter()
                    actions = self.alg.act(obs, privileged_obs)
                    action_time += time.perf_counter() - action_start

                    # Step the environment
                    env_start = time.perf_counter()
                    obs, rewards, dones, infos = self.env.step(
                        actions.to(self.env.device)
                    )
                    env_step_time += time.perf_counter() - env_start
                    
                    # Randomize episode length for reset environments to prevent synchronization
                    if self.randomize_reset_episode_progress > 0:
                        reset_env_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                        if len(reset_env_ids) > 0:
                            # Randomize episode progress for reset environments
                            max_progress = int(self.randomize_reset_episode_progress * self.env.max_episode_length)
                            random_lengths = torch.randint(
                                0, max_progress + 1, (len(reset_env_ids),),
                                device=self.device, dtype=torch.long
                            )
                            self.env.episode_length_buf[reset_env_ids] = random_lengths
                    
                    # Process observations and rewards
                    process_start = time.perf_counter()
                    # Move to device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(
                                self.device
                            )
                        )
                    else:
                        privileged_obs = obs

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)
                    process_time += time.perf_counter() - process_start

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.perf_counter()
                collection_time = stop - start
                # Store detailed timing
                simulation_time = env_step_time + action_time + process_time
                start = stop

                # compute returns
                self.alg.compute_returns(privileged_obs)

            # update policy
            if self.empirical_normalization:
                loss_dict = self.alg.update(
                    obs_normalizer=self.obs_normalizer,
                    privileged_obs_normalizer=self.privileged_obs_normalizer
                )
            else:
                loss_dict = self.alg.update()

            # Update EMA after PPO update (Option B: per PPO update)
            if self.alg.ema is not None:
                if self.alg.tot_timesteps == self.alg.ema_warmup_steps:
                    # At warmup threshold, reset EMA to current weights
                    self.alg.ema.reset_to_current()
                elif self.alg.tot_timesteps > self.alg.ema_warmup_steps:
                    # After warmup, do normal EMA updates
                    self.alg.ema.update()

            stop = time.perf_counter()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())

                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(
                os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )

        # Run post-training checkpoint evaluation
        if self.log_dir is not None and not self.disable_logs:
            self.run_post_training_checkpoint_eval()

        # Properly close the writer to ensure all metrics are flushed
        if self.writer is not None:
            if hasattr(self.writer, 'stop'):
                self.writer.stop()
            if hasattr(self.writer, 'close'):
                self.writer.close()

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = (
            self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        )
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            if key == "metrics":
                # Handle metrics dict (non-loss metrics like clip_param, grad norms, etc.)
                for metric_name, metric_value in value.items():
                    self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, locs["it"])
            else:
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        
        # -- Detailed timing breakdown
        if "env_step_time" in locs:
            self.writer.add_scalar("Perf/env_step_time", locs["env_step_time"], locs["it"])
            self.writer.add_scalar("Perf/action_time", locs["action_time"], locs["it"])
            self.writer.add_scalar("Perf/process_time", locs["process_time"], locs["it"])
            self.writer.add_scalar("Perf/simulation_time", locs["simulation_time"], locs["it"])
            
            # Calculate percentages
            total_time = locs["collection_time"] + locs["learn_time"]
            sim_percentage = (locs["simulation_time"] / total_time) * 100 if total_time > 0 else 0
            opt_percentage = (locs["learn_time"] / total_time) * 100 if total_time > 0 else 0
            
            self.writer.add_scalar("Perf/simulation_percentage", sim_percentage, locs["it"])
            self.writer.add_scalar("Perf/optimization_percentage", opt_percentage, locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {
                    locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                # Skip non-scalar values (like histograms)
                if isinstance(value, (int, float)):
                    log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            # -- Rewards
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # -- episode info
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {
                    locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
            )
            for key, value in locs["loss_dict"].items():
                # Skip non-scalar values (like histograms)
                if isinstance(value, (int, float)):
                    log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (
                            locs["start_iter"]
                            + locs["num_learning_iterations"]
                            - locs["it"]
                        )
                    ),
                )
            }\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Prepare model state dict (use EMA if available)
        model_state_dict = self.alg.policy.state_dict()
        if self.alg.ema is not None and self.alg.tot_timesteps > self.alg.ema_warmup_steps:
            # Replace actor weights with EMA shadow params.
            # EMA tracks policy.actor params (keys like "0.weight"), but
            # policy.state_dict() prefixes them with "actor." ("actor.0.weight").
            # Only do this after EMA warmup — before warmup, shadow params are
            # copies of random init weights, not the current trained weights.
            for name, ema_param in self.alg.ema.shadow_params.items():
                full_name = f"actor.{name}"
                if full_name in model_state_dict:
                    model_state_dict[full_name] = ema_param.clone()

        # -- Save model
        saved_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save EMA state if used
        if self.alg.ema is not None:
            saved_dict["ema_state_dict"] = self.alg.ema.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(
            loaded_dict["model_state_dict"]
        )
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(
                    loaded_dict["privileged_obs_norm_state_dict"]
                )
            else:
                # if the training is not resumed but a model is loaded, load the actor normalizer
                # for the privileged obs normalizer (observation space may differ)
                self.privileged_obs_normalizer.load_state_dict(
                    loaded_dict["obs_norm_state_dict"]
                )
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # -- Load EMA state if used
        if self.alg.ema is not None:
            if "ema_state_dict" in loaded_dict:
                self.alg.ema.load_state_dict(loaded_dict["ema_state_dict"])
                print("[INFO] Loaded EMA state from checkpoint")
            else:
                print("[WARNING] EMA is enabled but no EMA state found in checkpoint")
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            # -- restore common_step_counter to avoid restarting episode length warmup
            # Calculate total steps from iteration count
            steps_per_iteration = self.num_steps_per_env * self.env.num_envs
            total_steps = self.current_learning_iteration * steps_per_iteration
            # Set the environment's common_step_counter
            if hasattr(self.env.unwrapped, 'common_step_counter'):
                self.env.unwrapped.common_step_counter = total_steps
                print(f"[INFO] Restored common_step_counter to {total_steps} based on iteration {self.current_learning_iteration}")
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg.empirical_normalization:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def get_checkpoint_paths(self):
        """Scan log directory for checkpoint files and return sorted list.

        Returns:
            List of (iteration, filepath) tuples, filtered by post_eval_checkpoint_interval
        """
        import glob
        import re

        if self.log_dir is None:
            return []

        # Find all checkpoint files
        checkpoint_pattern = os.path.join(self.log_dir, "model_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)

        # Parse iteration numbers from filenames
        checkpoints = []
        for filepath in checkpoint_files:
            filename = os.path.basename(filepath)
            match = re.match(r"model_(\d+)\.pt", filename)
            if match:
                iteration = int(match.group(1))
                checkpoints.append((iteration, filepath))

        # Sort by iteration
        checkpoints.sort(key=lambda x: x[0])

        # Filter by checkpoint interval (take every Nth checkpoint)
        if self.post_eval_checkpoint_interval > 1:
            checkpoints = checkpoints[::self.post_eval_checkpoint_interval]

        return checkpoints

    def evaluate_checkpoint(self, checkpoint_path, iteration):
        """Evaluate a single checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            iteration: Iteration number of the checkpoint

        Returns:
            Dictionary containing evaluation metrics for each mode
        """
        import numpy as np

        # Initialize state variables to avoid NameError in exception handler
        current_model_state = None
        current_obs_norm_state = None
        current_priv_obs_norm_state = None

        try:
            # Save current model and normalizer state
            current_model_state = {k: v.clone() for k, v in self.alg.policy.state_dict().items()}
            if self.empirical_normalization:
                current_obs_norm_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                         for k, v in self.obs_normalizer.state_dict().items()}
                current_priv_obs_norm_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                              for k, v in self.privileged_obs_normalizer.state_dict().items()}

            # Load checkpoint (this loads both model and normalizer)
            loaded_dict = torch.load(checkpoint_path, weights_only=False)
            self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

            # Load normalizer state from checkpoint if available
            if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])

            # Note: post-warmup checkpoints have EMA weights baked into model_state_dict
            # (save() replaces actor weights with EMA shadow params after warmup).
            # Pre-warmup checkpoints have regular training weights (no EMA bake-in).
            # No need to do ema.store()/copy_to() here — that would overwrite the
            # checkpoint's weights with end-of-training shadow_params.

            # Switch to eval mode
            self.eval_mode()

            # Determine number of episodes for this eval
            num_episodes = getattr(self.cfg, "eval_episodes", 10)

            # Get eval modes from config
            eval_modes = getattr(self.cfg, "flow_eval_modes", ["zero", "fixed_seed", "random"])
            eval_fixed_seed = getattr(self.cfg, "flow_eval_fixed_seed", 12345)

            eval_results = {}

            for mode in eval_modes:
                # Reset environments for each mode
                obs, _ = self.env.reset()
                obs = obs.to(self.device)

                # Collect episodes for this mode
                mode_rewards = []
                mode_lengths = []

                # Track episode data
                episode_reward = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

                # Get max episode length
                max_episode_length = self.env.max_episode_length if hasattr(self.env, 'max_episode_length') else 1000

                # Run until we collect enough episodes
                episodes_per_env = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
                target_episodes_per_env = max(1, num_episodes // self.env.num_envs)

                while (episodes_per_env < target_episodes_per_env).any():
                    with torch.no_grad():
                        # Normalize observations if needed
                        if self.cfg.empirical_normalization:
                            norm_obs = self.obs_normalizer(obs)
                        else:
                            norm_obs = obs

                        # Use flow matching with specified evaluation mode
                        actions = self.alg.policy.act_inference(
                            norm_obs,
                            eval_mode=mode,
                            eval_fixed_seed=eval_fixed_seed
                        )

                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)

                    # Accumulate rewards and lengths
                    episode_reward += rewards.to(self.device)
                    episode_length += 1

                    # Check for done episodes
                    done_mask = (dones > 0) | (episode_length >= max_episode_length)

                    if done_mask.any():
                        # Record completed episodes
                        for idx in done_mask.nonzero(as_tuple=False).squeeze(-1):
                            if episodes_per_env[idx] < target_episodes_per_env:
                                mode_rewards.append(episode_reward[idx].item())
                                mode_lengths.append(episode_length[idx].item())
                                episodes_per_env[idx] += 1

                        # Reset the environments that are done
                        episode_reward[done_mask] = 0
                        episode_length[done_mask] = 0

                # Store results for this mode
                eval_results[mode] = {
                    "mean_reward": np.mean(mode_rewards) if mode_rewards else 0,
                    "std_reward": np.std(mode_rewards) if mode_rewards else 0,
                    "mean_length": np.mean(mode_lengths) if mode_lengths else 0,
                    "std_length": np.std(mode_lengths) if mode_lengths else 0,
                    "num_episodes": len(mode_rewards)
                }

            # Restore original model and normalizer state
            self.alg.policy.load_state_dict(current_model_state)
            if self.empirical_normalization:
                self.obs_normalizer.load_state_dict(current_obs_norm_state)
                self.privileged_obs_normalizer.load_state_dict(current_priv_obs_norm_state)

            # Switch back to train mode
            self.train_mode()

            # Clear GPU cache
            torch.cuda.empty_cache()

            return eval_results

        except Exception as e:
            print(f"Warning: Failed to evaluate checkpoint at iteration {iteration}: {e}")
            # Attempt to restore state even on failure
            try:
                if current_model_state is not None:
                    self.alg.policy.load_state_dict(current_model_state)
                if self.empirical_normalization and current_obs_norm_state is not None:
                    self.obs_normalizer.load_state_dict(current_obs_norm_state)
                    self.privileged_obs_normalizer.load_state_dict(current_priv_obs_norm_state)
                self.train_mode()
            except:
                pass
            return None

    def run_post_training_checkpoint_eval(self):
        """Run post-training evaluation on all saved checkpoints.

        Evaluates all checkpoints and logs results to WandB with custom eval_iteration metric.
        Only runs on rank 0 in distributed training.
        """
        # Skip if disabled or not on main rank
        if not self.enable_post_training_eval:
            return

        if self.disable_logs:
            print("[INFO] Skipping post-training eval on non-main rank in distributed training")
            return

        if self.writer is None:
            print("[WARNING] No writer available, skipping post-training eval")
            return

        print("\n" + "=" * 80)
        print("Starting post-training checkpoint evaluation...")
        print("=" * 80)

        # Get all checkpoint paths
        checkpoints = self.get_checkpoint_paths()

        if not checkpoints:
            print("[WARNING] No checkpoints found for post-training evaluation")
            return

        print(f"Found {len(checkpoints)} checkpoint(s) to evaluate")
        if self.post_eval_checkpoint_interval > 1:
            print(f"  (evaluating every {self.post_eval_checkpoint_interval} checkpoint(s))")

        num_episodes = getattr(self.cfg, "eval_episodes", 10)
        print(f"  Episodes per mode: {num_episodes}")
        print(f"  Eval modes: {getattr(self.cfg, 'flow_eval_modes', ['zero', 'fixed_seed', 'random'])}")

        # Evaluate each checkpoint
        all_results = {}
        for idx, (iteration, checkpoint_path) in enumerate(checkpoints, 1):
            print(f"\n[{idx}/{len(checkpoints)}] Evaluating checkpoint at iteration {iteration}...")

            eval_results = self.evaluate_checkpoint(checkpoint_path, iteration)

            if eval_results is None:
                print(f"  Skipped due to error")
                continue

            # Store results
            all_results[iteration] = eval_results

            # Log to wandb with custom x-axis
            if self.logger_type == "wandb":
                try:
                    import wandb
                    # Consolidate all metrics into a single log call
                    log_dict = {"eval_iteration": iteration}
                    for mode, results in eval_results.items():
                        log_dict[f"PostEval_{mode}/mean_reward"] = results["mean_reward"]
                        log_dict[f"PostEval_{mode}/std_reward"] = results["std_reward"]
                        log_dict[f"PostEval_{mode}/mean_episode_length"] = results["mean_length"]
                        log_dict[f"PostEval_{mode}/std_episode_length"] = results["std_length"]
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"  Warning: Failed to log to wandb: {e}")
            else:
                # For tensorboard/neptune, use regular writer (they don't have the step ordering issue)
                for mode, results in eval_results.items():
                    self.writer.add_scalar(f"PostEval_{mode}/mean_reward", results["mean_reward"], iteration)
                    self.writer.add_scalar(f"PostEval_{mode}/std_reward", results["std_reward"], iteration)
                    self.writer.add_scalar(f"PostEval_{mode}/mean_episode_length", results["mean_length"], iteration)
                    self.writer.add_scalar(f"PostEval_{mode}/std_episode_length", results["std_length"], iteration)

            # Print results for this checkpoint
            for mode, results in eval_results.items():
                print(f"    {mode:12s}: reward={results['mean_reward']:8.2f} ± {results['std_reward']:6.2f}, "
                      f"length={results['mean_length']:6.1f} ± {results['std_length']:5.1f}")

        # Print summary
        print("\n" + "=" * 80)
        print("Post-training checkpoint evaluation completed!")
        print(f"Successfully evaluated {len(all_results)}/{len(checkpoints)} checkpoints")
        print("=" * 80 + "\n")

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
