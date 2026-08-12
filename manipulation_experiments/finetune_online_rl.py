#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import logging
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict, Any

import imageio
import numpy as np
import torch
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from termcolor import colored
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
import tyro
import wandb

from lerobot.common.policies.pretrained import PreTrainedPolicy
from src.dexmg_env import VectorizedEnvWrapper, create_vectorized_env
from src.flow_model import FlowMatchingPolicy
from src.flow_model_config import FlowMatchingConfig

# ---- Multiprocessing start method (CUDA compat) ------------------------------
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---- Logging ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


@dataclass
class FlowPPOConfig:
    task: str = "Lift"
    loss_mode: Literal["fpo", "dppo"] = "fpo"
    observation_type: str = "image"
    num_envs: int = 2
    reset_every_iteration: bool = True
    truncation_as_done: bool = True
    camera_size: int = 84
    camera_name_to_vis: str = "agentview"

    # Base policy
    base_policy_wandb_project: Optional[str] = "flow-bc"
    base_policy_wandb_run_id: Optional[str] = None
    checkpoint_step: Optional[str] = "latest"
    base_policy_local_path: Optional[str] = None
    init_flow_network: bool = False
    load_ema: bool = True
    policy: str = "flowmatching"

    # Training
    total_timesteps: int = 1_000_000
    data_collection_steps: int = 96
    num_minibatches: int = 4
    update_epochs: int = 10
    gradient_accumulation_steps: int = 1
    n_iterations_train_only_value: int = 1

    # Policy overrides
    transported_clip_value: Optional[float] = None
    n_action_steps: Optional[int] = None
    sampling_steps: Optional[int] = None
    cfm_loss_use_huber: Optional[bool] = None
    cfm_loss_huber_delta: Optional[float] = None
    flow_network_output_param: Optional[Literal["u", "x0"]] = None
    cfm_loss_mode: Optional[Literal["u", "x0", "eps"]] = None
    image_observation_keys: Optional[str] = None  # space-separated -> list later

    # FPO
    freeze_vision_encoder: bool = True
    do_chunk_level_ppo: bool = True
    do_average_cfm_loss_in_chunk: bool = False
    n_action_samples: int = 16
    clamp_old_cfm_loss: Optional[float] = None
    cfm_loss_average_group_size: int = 1
    trust_region_mode: Literal["ppo", "spo", "aspo"] = "ppo"
    clamp_logratio: Optional[float] = None
    cfm_loss_weight_from_t: str = "constant"
    exploration_noise_std: Optional[float] = None
    zero_sampling: bool = True  # deprecated. now we evaluate with both zero and non zero sampling
    save_non_zero_sampling_video: bool = False

    # DPPO
    sde_sigma: float = 0.08
    learn_sde_sigma: bool = False
    noise_injection_min: float = 0.2
    noise_injection_max: float = 0.5
    entropy_loss_coef: float = 0.001
    dppo_norm_factor: float = 1.0
    average_logprob_over_denoising_steps: bool = False # something like reinflow style

    # PPO-ish
    discount: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.01
    spo_clip_coef: float = 0.01
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.1

    # LRs
    learning_rate_actor: float = 1e-5
    learning_rate_critic: float = 1e-4
    optimizer_betas_actor: list[float] = field(default_factory=lambda: [0.9, 0.99])

    # Schedulers
    lr_scheduler_name: str = "constant"
    lr_scheduler_actor_warmup_steps: int = 5
    lr_scheduler_critic_warmup_steps: int = 1

    # Eval
    rollout_freq: Optional[int] = 10
    eval_env: Optional[str] = None
    eval_num_episodes: int = 10
    eval_camera_size: int = 84
    eval_ema: bool = False

    # System
    seed: Optional[int] = None
    torch_deterministic: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_freq: int = 10
    save_freq: int = 100
    debug: bool = False
    distributed: bool = False

    # W&B
    wandb_enable: bool = True
    wandb_project: Optional[str] = "flow-bc-fpo-finetuning"
    wandb_entity: str = "far-wandb"
    wandb_notes: Optional[str] = None
    wandb_continue_run_id: Optional[str] = None
    experiment: str = "finetune_fpo"

    # Run
    output_dir: Optional[str] = None


# ---- Helper modules ----------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, global_obs_dim: int = 1033):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(global_obs)


@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + discount * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns


def clamp_ste(x, min=None, max=None):
    clamped = x.clamp(min=min, max=max)
    return x + (clamped - x).detach()


def _annotate_frame(
    frame: np.ndarray,
    env_idx: int,
    episode_num: int,
    total_episodes: int,
    episode_step: int,
    is_success: bool,
    font=None,
) -> np.ndarray:
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    episode_text = f"Env {env_idx + 1} | Episode {episode_num}/{total_episodes}"
    step_text = f"Step {episode_step}"
    status_text = "SUCCESS" if is_success else "FAIL"
    status_color = (0, 255, 0) if is_success else (255, 0, 0)
    y_offset = 10
    draw.text((10, y_offset), episode_text, fill=(255, 255, 255), font=font); y_offset += 15
    draw.text((10, y_offset), step_text, fill=(255, 255, 255), font=font); y_offset += 15
    draw.text((10, y_offset), status_text, fill=status_color, font=font)
    return np.array(pil_img)


def _run_rollouts(
    *,
    policy: PreTrainedPolicy,
    env: VectorizedEnvWrapper,
    save_dir: Path,
    global_step: int,
    num_episodes: int,
    task: str,
    zero_sampling: bool,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    was_training = policy.training
    policy.eval()

    num_parallel_envs = env.num_envs
    env_name = getattr(env, "env_name", "Unknown")
    logger.info(f"Running rollouts with environment: {env_name}")
    logger.info(f"Starting evaluation: {num_episodes} episodes using {num_parallel_envs} parallel environments")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_path = save_dir / f"eval_step_{global_step}_{now}.mp4"
    video_writer = imageio.get_writer(video_path.as_posix(), fps=20)

    obs, _ = env.reset()
    episode_frames = [[] for _ in range(num_parallel_envs)]
    episode_steps = [0] * num_parallel_envs

    policy.reset()

    successes_list = []
    done_episodes_list = []
    total_steps = 0
    t0 = time.perf_counter()

    while sum(done_episodes_list) < num_episodes:
        with torch.inference_mode():
            action, mdp_x_t_path = policy.select_action(obs, zero_sampling=zero_sampling, sde_sampling=False)

        obs, reward, terminated, truncated, info = env.step(action)
        frames = env.render()

        for env_idx in range(num_parallel_envs):
            episode_frames[env_idx].append(frames[env_idx])
            episode_steps[env_idx] += 1

        total_steps += num_parallel_envs
        done = terminated | truncated

        if any(done):
            terminated_envs = torch.where(done)[0]
            success_envs = torch.where(reward == 1.0)[0]
            policy.reset(env_ids=terminated_envs)

            for env_idx_tensor in terminated_envs:
                env_idx = int(env_idx_tensor.item())
                is_success = env_idx_tensor in success_envs
                done_episodes_list.append(1)
                successes_list.append(int(is_success))

                # discard the frames from the terminated environments since it is a new observation on the reseted new environment
                episode_frames[env_idx].pop(-1)
                episode_steps[env_idx] -= 1

                for step_idx, frame in enumerate(episode_frames[env_idx]):
                    annotated_frame = _annotate_frame(
                        frame=frame,
                        env_idx=env_idx,
                        episode_num=sum(done_episodes_list),
                        total_episodes=num_episodes,
                        episode_step=step_idx + 1,
                        is_success=is_success,
                        font=font,
                    )
                    video_writer.append_data(annotated_frame)

                episode_frames[env_idx] = []
                episode_steps[env_idx] = 0

        if total_steps % 1_000 == 0:
            logger.info(
                f"Total steps: {total_steps}, done episodes: {sum(done_episodes_list)}, successes: {sum(successes_list)}, "
                f"FPS: {total_steps / (time.perf_counter() - t0):.1f}"
            )
    
    # trim the tails
    successes = sum(successes_list[:num_episodes])
    done_episodes = sum(done_episodes_list[:num_episodes])

    video_writer.close()
    success_rate = successes / done_episodes if done_episodes > 0 else 0.0

    if was_training:
        policy.train()

    elapsed = time.perf_counter() - t0
    fps = total_steps / elapsed if elapsed > 0 else 0.0
    eps_sec = done_episodes / elapsed if elapsed > 0 else 0.0

    logger.info(f"Evaluation completed: {done_episodes} episodes, {successes} successes ({success_rate * 100:.1f}%)")
    logger.info(f"Performance: {total_steps} total steps in {elapsed:.1f}s")
    logger.info(f"Average FPS: {fps:.1f} frames/sec | Episodes/sec: {eps_sec:.2f}")
    logger.info(f"Video saved: {video_path}")

    return success_rate, video_path, fps, int(successes), int(done_episodes)

def eval_all_ranks(
    *,
    env: VectorizedEnvWrapper,
    num_envs_per_process,
    actor_ddp_or_single,
    world_size: int,
    rank: int,
    is_ddp: bool,
    device_str: str,
    device,
    run_dir: Path,
    cfg: FlowPPOConfig,
    create_env_fn,
    global_step: int,
):
    """Run evaluation on every rank, reduce results, log from rank 0 only."""
    # Unwrap DDP for deepcopy
    base_actor = actor_ddp_or_single.module if is_ddp else actor_ddp_or_single
    eval_actor = copy.deepcopy(base_actor).to(device)
    eval_actor.eval()

    # Optional EMA for eval
    if cfg.eval_ema and hasattr(eval_actor, "enable_ema"):
        eval_actor.enable_ema()

    # Split total episodes across ranks (as balanced as possible)
    total_eps = cfg.eval_num_episodes
    base = total_eps // world_size
    rem = total_eps % world_size
    eps_this_rank = base + (1 if rank < rem else 0)
    if eps_this_rank == 0:
        eps_this_rank = 0  # some ranks may skip if episodes < world_size

    # Each rank can also choose its own num_envs for eval (keep small)
    num_envs_eval_rank = num_envs_per_process #min(cfg.eval_num_envs, max(1, eps_this_rank)) if eps_this_rank > 0 else 1

    # Buffers for local results
    local_successes = 0
    local_episodes  = 0
    local_fps_sum   = 0.0  # (optional) aggregate FPS across ranks
    local_video_path = None

    if eps_this_rank > 0:
        # Only rank 0 records a video to avoid N videos
        save_video = (rank == 0)
        eval_actor.init_action_buffers(num_envs_eval_rank)
        # Eval with zero sampling
        sr, video_path, fps, succ, eps = _run_rollouts(
            policy=eval_actor,
            env=env,
            save_dir=(run_dir / "videos") if save_video else (run_dir / "_scratch_no_video"),
            global_step=global_step,
            num_episodes=eps_this_rank,
            task=cfg.eval_env,
            zero_sampling=True,
        )
        local_successes = succ
        local_episodes  = eps
        local_fps_sum   = fps * eps  # weight fps by episodes
        local_video_path = video_path if save_video else None

        # Eval with non zero sampling
        sr_non_zero, video_path_non_zero, fps_non_zero, succ_non_zero, eps_non_zero = _run_rollouts(
            policy=eval_actor,
            env=env,
            save_dir=(run_dir / "videos") if save_video else (run_dir / "_scratch_no_video"),
            global_step=global_step,
            num_episodes=eps_this_rank,
            task=cfg.eval_env,
            zero_sampling=False,
        )
        local_successes_non_zero = succ_non_zero
        local_episodes_non_zero = eps_non_zero
        local_fps_sum_non_zero = fps_non_zero * eps_non_zero
        local_video_path_non_zero = video_path_non_zero if save_video else None


    # Reduce across ranks
    if is_ddp:
        t = torch.tensor(
            [local_successes, local_episodes, local_fps_sum, local_successes_non_zero, local_episodes_non_zero, local_fps_sum_non_zero],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        global_successes, global_episodes, global_fps_sum, global_successes_non_zero, global_episodes_non_zero, global_fps_sum_non_zero = t.tolist()
    else:
        global_successes, global_episodes, global_fps_sum, global_successes_non_zero, global_episodes_non_zero, global_fps_sum_non_zero = (
            float(local_successes), float(local_episodes), float(local_fps_sum), float(local_successes_non_zero), float(local_episodes_non_zero), float(local_fps_sum_non_zero)
        )

    assert global_episodes == global_episodes_non_zero, "global_episodes and global_episodes_non_zero should be the same"


    # Compute global metrics on rank 0
    if rank == 0:
        global_sr = (global_successes / max(1.0, global_episodes))
        global_sr_non_zero = (global_successes_non_zero / max(1.0, global_episodes_non_zero))
        # episode-weighted average FPS (approx)
        global_fps = (global_fps_sum / max(1.0, global_episodes))
        global_fps_non_zero = (global_fps_sum_non_zero / max(1.0, global_episodes_non_zero))

        if cfg.save_non_zero_sampling_video:
            local_video_path = local_video_path_non_zero

        return {
            "success_rate": global_sr,
            "success_rate_non_zero": global_sr_non_zero,
            "fps": global_fps,
            "fps_non_zero": global_fps_non_zero,
            "episodes": int(global_episodes),
            "video_path": local_video_path,  # only rank 0 produced one
        }
    return None


def save_checkpoint(checkpoint_dir: Path, step: int, policy: PreTrainedPolicy, optimizer):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(checkpoint_dir / "policy"))
    checkpoint_data = {"step": step, "optimizer_state_dict": optimizer.state_dict()}
    if hasattr(policy, "ema_model") and getattr(policy, "ema_model", None) is not None:
        checkpoint_data["ema_state_dict"] = policy.ema_model.state_dict()
    torch.save(checkpoint_data, checkpoint_dir / "optimizer.pt")


# ---- W&B artifact pull (rank 0 only) ----------------------------------------
def download_checkpoint_from_wandb_rank0(
    run_id: str,
    project: str,
    entity: str,
    artifact_alias: str = "latest",
    download_dir: Path = Path("./downloaded_checkpoints"),
) -> Path:
    """Download a checkpoint artifact from wandb."""
    logger.info(colored(f"Downloading checkpoint from W&B run {run_id} (artifact: {artifact_alias})...", "cyan"))

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    logger.info(colored(f"Fetching artifacts from run {run_id}...", "cyan"))
    artifacts = run.logged_artifacts()

    if not artifacts:
        raise ValueError(f"No artifacts found for run {run_id}")

    logger.info(colored(f"Found {len(artifacts)} total artifacts in run {run_id}", "cyan"))

    # Filter for checkpoint artifacts only
    checkpoint_artifacts = [a for a in artifacts if "checkpoint" in a.name]

    if not checkpoint_artifacts:
        raise ValueError(f"No checkpoint artifacts found for run {run_id}")

    logger.info(colored(f"Found {len(checkpoint_artifacts)} checkpoint artifacts", "cyan"))

    def get_step_from_artifact(artifact):
        """Extract step number from artifact metadata or name."""
        if hasattr(artifact, 'metadata') and artifact.metadata and 'step' in artifact.metadata:
            return artifact.metadata['step']
        import re
        match = re.search(r'step_(\d+)', artifact.name)
        if match:
            return int(match.group(1))
        return 0

    # Find the right artifact based on alias
    artifact = None

    if artifact_alias == "latest":
        checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
        artifact = checkpoint_artifacts[0]
        logger.info(colored(f"Looking for latest checkpoint...", "cyan"))
    elif artifact_alias == "best":
        best_artifacts = [a for a in checkpoint_artifacts if "best" in a.name.lower()]
        if best_artifacts:
            artifact = best_artifacts[0]
            logger.info(colored(f"Found 'best' checkpoint", "cyan"))
        else:
            logger.warning(colored(f"No 'best' checkpoint found, falling back to latest", "yellow"))
            checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
            artifact = checkpoint_artifacts[0]
    else:
        # Try to find by specific step or name
        matching_artifacts = [a for a in checkpoint_artifacts if artifact_alias in a.name.lower()]
        if matching_artifacts:
            artifact = matching_artifacts[0]
            logger.info(colored(f"Found checkpoint matching '{artifact_alias}'", "cyan"))
        else:
            logger.warning(colored(f"No checkpoint matching '{artifact_alias}' found, falling back to latest", "yellow"))
            checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
            artifact = checkpoint_artifacts[0]

    step_num = get_step_from_artifact(artifact)
    logger.info(colored(f"Selected artifact: {artifact.name} (step {step_num})", "green"))
    logger.info(colored(f"Available checkpoints (sorted by step):", "cyan"))

    # Show all available checkpoints
    checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
    for i, a in enumerate(checkpoint_artifacts[:10]):
        step = get_step_from_artifact(a)
        selected_marker = "✓" if a == artifact else " "
        logger.info(colored(f"  {selected_marker} {i+1}. {a.name} (step {step})", "cyan"))

    # Download the artifact
    download_path = download_dir / f"{run_id}_{artifact_alias}"
    artifact_dir = artifact.download(root=str(download_path))

    logger.info(colored(f"Checkpoint downloaded to: {artifact_dir}", "green"))
    return Path(artifact_dir)


# ---- Main --------------------------------------------------------------------
def main(cfg: FlowPPOConfig):
    # In case you train a policy from scratch with rewards
    # Normalize image_observation_keys -> list[str] (or None)
    cfg.image_observation_keys = (
        cfg.image_observation_keys.split(" ") if cfg.image_observation_keys is not None else None
    )

    # ---------------- DDP Setup ----------------
    is_ddp = cfg.distributed
    if is_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        # Rely on env vars; do NOT pass rank/world_size manually
        dist.init_process_group(backend="nccl")
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(cfg.device)

    logger.info(colored(f"[{rank}/{world_size}] Using device: {device}", "green"))

    # Run dir on all ranks (avoid races)
    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{cfg.experiment}_{run_start_time}" if cfg.output_dir is None else Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()
    if rank == 0:
        logger.info(colored(f"Run directory: {run_dir}", "green"))

    # Seeding
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    if rank == 0:
        logger.info(colored(f"Random seed set to {cfg.seed}", "yellow"))

    # ------------- Rank-0 I/O -> broadcast config/weights ----------------
    policy_config_payload: Dict[str, Any] | None = None
    state_dict_payload: Dict[str, Any] | None = None
    ema_state_payload: Dict[str, Any] | None = None

    if rank == 0:
        # Decide policy directory or load artifacts
        if cfg.base_policy_wandb_run_id is not None:
            alias = cfg.checkpoint_step if cfg.checkpoint_step is not None else "latest"
            if cfg.wandb_project is None:
                raise ValueError("--wandb_project is required when using --base_policy_wandb_run_id")
            ckpt_root = download_checkpoint_from_wandb_rank0(
                run_id=cfg.base_policy_wandb_run_id,
                project=cfg.base_policy_wandb_project,
                entity=cfg.wandb_entity,
                artifact_alias=alias,
                download_dir=run_dir / "downloaded_checkpoints",
            )
            policy_dir = ckpt_root / "policy"
        elif cfg.base_policy_local_path is not None:
            pd = Path(cfg.base_policy_local_path) / "policy"
            policy_dir = pd if pd.exists() else Path(cfg.base_policy_local_path)
        else:
            raise ValueError("Must provide either --base_policy_wandb_run_id or --base_policy_local_path")

        logger.info(colored(f"[Global] Policy directory: {policy_dir}", "cyan"))
        config_path = policy_dir / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        # sanitize for FlowMatchingConfig
        config_dict.pop("type", None)
        config_dict.pop("normalization_mapping", None)
        policy_config_payload = config_dict

        weights_path = policy_dir / "model.safetensors"
        if not weights_path.exists():
            raise ValueError(f"Model weights file not found: {weights_path}")
        state_dict_payload = load_file(weights_path, device="cpu")

        # Load EMA from optimizer.pt if available
        optimizer_path = policy_dir.parent / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_blob = torch.load(optimizer_path, map_location="cpu")
            ema_state_payload = optimizer_blob.get("ema_state_dict", None)

    # Broadcast payloads
    if is_ddp:
        obj = [policy_config_payload, state_dict_payload, ema_state_payload]
        dist.broadcast_object_list(obj, src=0)
        policy_config_payload, state_dict_payload, ema_state_payload = obj

    # ---------- Build actor/critic identically on every rank --------------
    # Construct FlowMatchingConfig from payload
    policy_config = FlowMatchingConfig(**policy_config_payload)  # type: ignore[arg-type]

    # Derive features from input_features (same as pretraining)
    policy_config.image_features = [k for k in policy_config.input_features if "image" in k]
    policy_config.state_features = [k for k in policy_config.input_features if "state" in k or "pos" in k]
    logger.info(f"[Rank {rank}] Image features: {policy_config.image_features}")
    logger.info(f"[Rank {rank}] State features: {policy_config.state_features}")

    # Update cfg.image_observation_keys from config fnames
    cfg.image_observation_keys = [k.replace("observation.images.", "") for k in policy_config.image_features]
    logger.info(f"[Rank {rank}] Using image features from checkpoint config: {policy_config.image_features}")

    # Instantiate actor
    logger.info(colored(f"[Rank {rank}] Constructing FlowMatchingPolicy from config...", "cyan"))
    actor = FlowMatchingPolicy(policy_config, dataset_stats=None)

    # Load weights from broadcast payload
    actor.load_state_dict(state_dict_payload, strict=True)  # type: ignore[arg-type]
    logger.info(colored(f"[Rank {rank}] Loaded policy from checkpoint", "green"))

    # Apply EMA if provided + requested
    if getattr(actor, "ema_model", None) is not None and ema_state_payload is not None:
        actor.ema_model.load_state_dict(ema_state_payload)  # type: ignore[arg-type]
        if cfg.load_ema:
            actor.ema_model.copy_to(actor.model.parameters())
            if rank == 0:
                logger.info(colored("Loaded and applied EMA weights from checkpoint", "green"))
    elif cfg.load_ema and rank == 0:
        logger.warning(colored("--load_ema flag set but no EMA weights found", "yellow"))


    # --------- Apply policy overrides (must be identical across ranks) ----
    def log_override(name, new, old):
        logger.info(f"[Rank {rank}] Overriding {name} to {new} from base policy {old}")

    if cfg.n_action_steps is not None:
        # Only override if not already set in the base policy config
        logger.warning(f"Only if you are training from scratch, otherwise you should use the same n_action_steps as the base policy ({actor.config.n_action_steps})")
        log_override("n_action_steps", cfg.n_action_steps, actor.config.n_action_steps)
        actor.config.n_action_steps = cfg.n_action_steps
    else:
        cfg.n_action_steps = actor.config.n_action_steps

    if cfg.sampling_steps is not None:
        log_override("sampling_steps", cfg.sampling_steps, actor.config.sampling_steps)
        actor.config.sampling_steps = cfg.sampling_steps

    if cfg.init_flow_network:
        logger.info(colored("[Rank {rank}] Initialize flow action network to random weights", "cyan"))
        actor.model.initialize_layers()

    if cfg.cfm_loss_use_huber is not None:
        log_override("cfm_loss_use_huber", cfg.cfm_loss_use_huber, actor.config.cfm_loss_use_huber)
        actor.config.cfm_loss_use_huber = cfg.cfm_loss_use_huber

    if cfg.cfm_loss_huber_delta is not None:
        log_override("cfm_loss_huber_delta", cfg.cfm_loss_huber_delta, actor.config.cfm_loss_huber_delta)
        actor.config.cfm_loss_huber_delta = cfg.cfm_loss_huber_delta

    if cfg.flow_network_output_param is not None:
        log_override("flow_network_output_param", cfg.flow_network_output_param, actor.config.flow_network_output_param)
        actor.config.flow_network_output_param = cfg.flow_network_output_param

    if cfg.cfm_loss_mode is not None:
        log_override("cfm_loss_mode", cfg.cfm_loss_mode, actor.config.cfm_loss_mode)
        actor.config.cfm_loss_mode = cfg.cfm_loss_mode
    
    if cfg.transported_clip_value is not None:
        log_override("transported_clip_value", cfg.transported_clip_value, actor.config.transported_clip_value)
        actor.config.transported_clip_value = cfg.transported_clip_value

    if cfg.cfm_loss_weight_from_t is not None:
        log_override("cfm_loss_weight_from_t", cfg.cfm_loss_weight_from_t, actor.config.cfm_loss_weight_from_t)
        actor.config.cfm_loss_weight_from_t = cfg.cfm_loss_weight_from_t

    if cfg.exploration_noise_std is not None:
        logger.info(f"[Rank {rank}] Overriding exploration_noise_std to {cfg.exploration_noise_std} "
                    f"from base policy {getattr(actor, 'exploration_noise_std', None)}")
        actor.exploration_noise_std = cfg.exploration_noise_std

    if cfg.sde_sigma is not None:
        logger.info(f"[Rank {rank}] Overriding sde_sigma to {cfg.sde_sigma} in actor config "
                    f"from base policy {getattr(actor, 'config.sde_sigma', None)}")
        actor.config.sde_sigma = cfg.sde_sigma
    if cfg.learn_sde_sigma:
        logger.info(f"[Rank {rank}] Overriding learn_sde_sigma to {cfg.learn_sde_sigma} in actor config "
                    f"from base policy {getattr(actor, 'config.learn_sde_sigma', None)}")
        actor.config.learn_sde_sigma = cfg.learn_sde_sigma
        actor.config.noise_injection_min = cfg.noise_injection_min
        actor.config.noise_injection_max = cfg.noise_injection_max
        actor.initialize_noise_injection_network()

    assert actor.config.n_action_steps == cfg.n_action_steps
    assert cfg.n_action_steps <= actor.config.horizon, \
        f"n_action_steps ({cfg.n_action_steps}) must be <= horizon ({actor.config.horizon})"

    # Critic
    critic = Critic(global_obs_dim=actor.model.global_cond_dim)

    # Move to device BEFORE DDP
    actor.to(device)
    if getattr(actor, "ema_model", None) is not None:
        actor.ema_model.to(device)
    critic.to(device)

    # Freeze vision encoder AFTER wrapping with DDP (same on all ranks)
    if cfg.freeze_vision_encoder:
        logger.info(colored(f"[Rank {rank}] Freezing vision encoder", "cyan"))
        for p in actor.model.vision_encoder.parameters():
            p.requires_grad = False

    # Wrap with DDP
    if is_ddp:
        # Remove ema model from actor before wrapping to avoid unused parameter issues
        actor.ema_model = None
        cfg.eval_ema = False
        logger.info(colored(f"[Rank {rank}] Removing EMA model for DDP compatibility", "cyan"))

        actor = DDP(
            actor,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        critic = DDP(
            critic,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )


    logger.info(f"[Rank {rank}] Actor: {actor.__class__.__name__}")
    actor_module = actor.module if is_ddp else actor
    logger.info(f"[Rank {rank}] N action steps: {actor_module.config.n_action_steps}, "
                f"prediction horizon: {actor_module.config.horizon}")
    logger.info(f"[Rank {rank}] Critic: {critic}")


    # ----------------- Environment setup -----------------
    n_action_samples = cfg.n_action_samples
    steps_per_iteration = cfg.data_collection_steps
    n_action_steps = cfg.n_action_steps

    group_size = n_action_samples if cfg.cfm_loss_average_group_size == -1 else cfg.cfm_loss_average_group_size
    n_groups = n_action_samples // group_size
    assert actor_module.config.horizon >= n_action_steps
    assert n_action_samples % group_size == 0
    assert steps_per_iteration % n_action_steps == 0
    
    if is_ddp:
        num_envs_per_process = cfg.num_envs // world_size + (1 if rank < (cfg.num_envs % world_size) else 0)
        batch_size = cfg.data_collection_steps * cfg.num_envs // n_action_steps
        local_batch_size = cfg.data_collection_steps * num_envs_per_process // n_action_steps
        minibatch_size = max(local_batch_size // cfg.num_minibatches, 1)
        logger.info(f"[Rank {rank}] Handling {num_envs_per_process} envs | "
                    f"Local batch: {local_batch_size} | Minibatch: {minibatch_size}")
    else:
        num_envs_per_process = cfg.num_envs
        batch_size = cfg.data_collection_steps * cfg.num_envs // n_action_steps
        local_batch_size = batch_size
        minibatch_size = max(batch_size // cfg.num_minibatches, 1)

    num_iterations = cfg.total_timesteps // batch_size
    assert cfg.gradient_accumulation_steps <= cfg.num_minibatches
    effective_minibatch_size = minibatch_size * cfg.gradient_accumulation_steps
    logger.info(f"[Rank {rank}] Grad accum steps: {cfg.gradient_accumulation_steps} | "
                f"Effective MB: {effective_minibatch_size} (base: {minibatch_size})")

    device_str = "cpu" if cfg.device == "cpu" else "cuda"
    env = create_vectorized_env(
        env_name=cfg.task,
        num_envs=num_envs_per_process,
        device=device_str,
        camera_size=cfg.camera_size,
        video_key="agentview",
        debug=cfg.debug,
        expected_image_keys=cfg.image_observation_keys,
    )
    # Init action buffers
    actor_module.init_action_buffers(num_envs_per_process)
    logger.info(colored(f"[Rank {rank}] Initialized action buffers for {num_envs_per_process} environments", "green"))

    # Obs/action dims
    joint_pos_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_space.shape[1]
    image_keys = actor_module.config.image_features
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    n_images = len(image_keys)
    logger.info(f"[Rank {rank}] Obs state dim: {joint_pos_dim} | Action dim: {action_dim} | "
                f"Image keys: {image_keys} | Image shape: ({img_c},{img_h},{img_w}) | n_images={n_images}")

    # ----------------- Optimizers / sched ----------------
    params_actor = actor.parameters()
    optimizer_actor = optim.AdamW(
        params_actor,
        lr=cfg.learning_rate_actor,
        betas=tuple(cfg.optimizer_betas_actor),
        eps=1e-5,
        weight_decay=1e-6,
    )
    lr_scheduler_actor = get_scheduler(
        name=cfg.lr_scheduler_name,
        optimizer=optimizer_actor,
        num_warmup_steps=cfg.lr_scheduler_actor_warmup_steps,
        num_training_steps=num_iterations,
    )
    critic_params = critic.parameters()
    optimizer_critic = optim.AdamW(critic_params, lr=cfg.learning_rate_critic, eps=1e-5, weight_decay=1e-6)
    lr_scheduler_critic = get_scheduler(
        name=cfg.lr_scheduler_name,
        optimizer=optimizer_critic,
        num_warmup_steps=cfg.lr_scheduler_critic_warmup_steps,
        num_training_steps=num_iterations,
    )
    logger.info(f"[Rank {rank}] Total timesteps: {cfg.total_timesteps}, batch size: {batch_size} | "
                f"MB: {minibatch_size}, iterations: {num_iterations}")

    # ----------------- W&B (rank 0 only) -----------------
    if cfg.wandb_enable and rank == 0:
        if cfg.wandb_project is None:
            raise ValueError("--wandb_project is required when --wandb_enable")
        wandb_run_id = cfg.wandb_continue_run_id if cfg.wandb_continue_run_id else None
        wandb_resume_mode = "must" if cfg.wandb_continue_run_id else None
        wandb_config = {**vars(cfg), "num_iterations": num_iterations, "batch_size": batch_size, "minibatch_size": minibatch_size}
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=wandb_config,
            name=f"{cfg.experiment}_{cfg.policy}_{cfg.task}",
            id=wandb_run_id,
            resume=wandb_resume_mode,
            dir=str(run_dir),
            settings=wandb.Settings(),
        )
        logger.info(colored(f"W&B logging enabled (logs saved to {run_dir / 'wandb'})", "blue"))

    # ----------------- Training storage ------------------
    checkpoints_dir = run_dir / "checkpoints" if rank == 0 else None
    if rank == 0:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0  # maintain on rank 0 only
    iteration = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0

    obs_state_stored = torch.zeros((steps_per_iteration, num_envs_per_process, joint_pos_dim))
    actions_stored = torch.zeros((steps_per_iteration, num_envs_per_process, action_dim))
    mdp_x_t_paths_stored = torch.zeros((steps_per_iteration, actor_module.config.sampling_steps, num_envs_per_process, action_dim))
    rewards_stored = torch.zeros((steps_per_iteration, num_envs_per_process))
    dones_stored = torch.zeros((steps_per_iteration, num_envs_per_process))
    values_stored = torch.zeros((steps_per_iteration, num_envs_per_process))
    cfm_losses_stored = torch.zeros((steps_per_iteration, num_envs_per_process, n_action_samples))
    cfm_loss_ts_stored = torch.zeros((steps_per_iteration, num_envs_per_process, n_action_samples))
    cfm_loss_epsilons_stored = torch.zeros((steps_per_iteration, num_envs_per_process, n_action_samples, action_dim))
    cfm_value_invalid_stored = torch.zeros((steps_per_iteration, num_envs_per_process))
    dppo_log_probs_stored = torch.zeros((steps_per_iteration, actor_module.config.sampling_steps, num_envs_per_process))

    next_done = torch.zeros(num_envs_per_process)
    next_obs, _ = env.reset()
    actor_module.reset()

    obs_images_stored_list: Dict[str, torch.Tensor] = {
        k: torch.zeros((steps_per_iteration, num_envs_per_process, 3, img_h, img_w))
        for k in next_obs.keys() if k.startswith("observation.images.")
    }

    def get_cfm_values(actor, obs, n_action_samples=1, cfm_loss_ts=None, cfm_loss_epsilons=None, debug=False):
        cfm_loss, cfm_loss_t, cfm_loss_eps = actor(
            obs, n_action_samples, cfm_loss_ts, cfm_loss_epsilons, debug=debug
        )
        return cfm_loss, cfm_loss_t, cfm_loss_eps

    def get_log_prob_and_entropy(actor, obs):
        log_prob, entropy, sde_sigma = actor(obs, is_dppo=True)

        return log_prob, entropy, sde_sigma

    def get_action_and_value(actor_module, critic, obs, sde_sampling: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_copy = copy.deepcopy(obs)
        action, mdp_x_t_path = actor_module.select_action(obs, sde_sampling=sde_sampling)
        obs_copy = actor_module.normalize_inputs(obs_copy)
        obs_cond = actor_module.model.encode_observations(obs_copy)
        value = critic(obs_cond)
        return action, mdp_x_t_path, value

    # ----------------- Training loop ---------------------
    logger.info(colored(f"[Rank {rank}] Starting the main training loop", "green"))

    while (iteration * cfg.data_collection_steps * cfg.num_envs) < cfg.total_timesteps:
        iteration += 1
        if rank == 0:
            logger.info(colored(
                f"Iteration: {iteration}/{num_iterations} | "
                f"Global step (approx): {global_step}/{cfg.total_timesteps}", "yellow"
            ))

        if cfg.reset_every_iteration:
            next_obs, _ = env.reset()
            actor_module.reset()

        done_episodes = 0
        successes = 0
        step = 0
        iteration_start_time = time.time()

        actor.eval()
        critic.eval()
        logger.info(f"[Rank {rank}] Starting data collection...")

        while step < steps_per_iteration:
            with torch.inference_mode():
                action_idx = 0
                first_obs_from_chunk = None

                while action_idx < n_action_steps and step < steps_per_iteration:
                    if first_obs_from_chunk is None:
                        first_obs_from_chunk = copy.deepcopy(next_obs)

                    dones_stored[step] = next_done
                    curr_obs = next_obs
                    sde_sampling = cfg.loss_mode == "dppo"
                    action, mdp_x_t_path, value = get_action_and_value(actor_module, critic, curr_obs, sde_sampling=sde_sampling)
                    assert mdp_x_t_path.shape == (num_envs_per_process, actor_module.config.sampling_steps, action_dim), \
                        f"mdp_x_t_path shape should be (num_envs_per_process, actor_module.config.sampling_steps, action_dim), but got {mdp_x_t_path.shape}"

                    next_obs, reward, next_done, truncated, _ = env.step(action)
                    if cfg.truncation_as_done:
                        next_done = next_done | truncated

                    for obs_key, obs_value in curr_obs.items():
                        if obs_key.startswith("observation.images."):
                            obs_images_stored_list[obs_key][step] = obs_value.cpu()
                    obs_state_stored[step] = curr_obs["observation.state"].cpu()

                    values_stored[step] = value.flatten().cpu()
                    actions_stored[step] = action.cpu()
                    mdp_x_t_paths_stored[step] = mdp_x_t_path.permute(1, 0, 2).cpu()
                    rewards_stored[step] = reward.view(-1).cpu()
                    next_done = next_done.view(-1).cpu()

                    if any(next_done):
                        done_episodes += next_done.sum().item()
                        successes += reward[torch.where(next_done)[0]].sum().item()
                        actor_module.reset(env_ids=torch.where(next_done)[0])

                    if step > 0 and step % 100 == 0:
                        sps_local = step * num_envs_per_process / (time.time() - iteration_start_time + 1e-9)
                        if done_episodes > 0:
                            success_rate = successes / done_episodes
                            msg_sr = f"{success_rate:.2%} from {int(done_episodes)} episodes"
                        else:
                            msg_sr = "0.00% from 0 episodes"
                        logger.info(f"[Rank {rank}] step={step}/{steps_per_iteration}, "
                                    f"sps_local={sps_local:.2f}, SR={msg_sr}")

                    action_idx += 1
                    step += 1

                    # Only rank 0 advances a global_step approximation
                    if rank == 0:
                        global_step += cfg.num_envs

                # Get CFM values for the chunk
                actor_module.reset()
                next_done[:] = False

                curr_obs_chunk = first_obs_from_chunk
                curr_obs_chunk["action"] = actions_stored[step - action_idx:step].permute(1, 0, 2).to(device)

                # actions_stored
                # curr_obs_chunk["action"] = actions_stored[step - horizon:step] 
                
                # For FPO finetuning
                cfm_loss, cfm_loss_t, cfm_loss_eps = get_cfm_values(
                    actor, curr_obs_chunk, n_action_samples
                )
                cfm_losses_stored[step - action_idx:step] = cfm_loss.cpu()
                cfm_loss_ts_stored[step - action_idx:step] = cfm_loss_t.cpu()
                cfm_loss_epsilons_stored[step - action_idx:step] = cfm_loss_eps.cpu()

                # For DPPO finetuning
                # mdp_x_t_paths_stored: (n_action_steps, sampling_steps, num_envs_per_process, action_dim) 
                # ->: (num_envs_per_process, n_action_steps, sampling_steps, action_dim)
                curr_obs_chunk["mdp_x_t_path"] = mdp_x_t_paths_stored[step - action_idx:step].permute(2, 0, 1, 3).to(device)
                dppo_log_prob, dppo_entropy, dppo_sde_sigma = get_log_prob_and_entropy(actor, curr_obs_chunk) # dppo_log_prob: (num_envs, horizon, sampling_steps)
                dppo_log_probs_stored[step - action_idx:step] = dppo_log_prob.permute(1, 2, 0).cpu()
                # dppo_entropy and dppo_sde_sigma are not stored during data collection, only used during training

                # Should be used both for FPO and DPPO finetuning
                chunk_dones = dones_stored[step - action_idx:step]
                cfm_value_invalid_chunk = cfm_value_invalid_stored[step - action_idx:step]
                for i in range(num_envs_per_process):
                    chunk_done_idx = torch.where(chunk_dones[:, i] == 1)[0]
                    if len(chunk_done_idx) > 0:
                        cfm_value_invalid_chunk[chunk_done_idx:, i] = 1
                cfm_value_invalid_stored[step - action_idx:step] = cfm_value_invalid_chunk

        # Local SR for this rank
        success_rate_local = (successes / done_episodes) if done_episodes > 0 else 0.0
        sps_local_total = steps_per_iteration * num_envs_per_process / (time.time() - iteration_start_time + 1e-9)

        # Optionally reduce SR stats to rank 0 (not strictly necessary for training)
        if is_ddp:
            t = torch.tensor([successes, done_episodes], dtype=torch.float32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            successes_global, episodes_global = t.tolist()
            success_rate_global = (successes_global / episodes_global) if episodes_global > 0 else 0.0
        else:
            success_rate_global = success_rate_local

        if rank == 0:
            logger.info(
                f"[Rank {rank}] SR: {success_rate_global:.2%} from {int(episodes_global)} episodes | SPS_local(rank0 est): {sps_local_total:.2f}"
            )

        # ---------- Reshape for training ----------
        b_actions = actions_stored.reshape(-1, n_action_steps, num_envs_per_process, action_dim)
        b_actions = b_actions.permute(0, 2, 1, 3).reshape(-1, n_action_steps, action_dim)

        b_cfm_losses = cfm_losses_stored.reshape(-1, n_action_steps, num_envs_per_process, n_action_samples)
        b_cfm_losses = b_cfm_losses.permute(0, 2, 1, 3).reshape(-1, n_action_steps, n_action_samples)

        b_cfm_loss_ts = cfm_loss_ts_stored.reshape(-1, n_action_steps, num_envs_per_process, n_action_samples)
        b_cfm_loss_ts = b_cfm_loss_ts.permute(0, 2, 1, 3).reshape(-1, n_action_steps, n_action_samples)

        b_cfm_loss_epsilons = cfm_loss_epsilons_stored.reshape(-1, n_action_steps, num_envs_per_process, n_action_samples, action_dim)
        b_cfm_loss_epsilons = b_cfm_loss_epsilons.permute(0, 2, 1, 3, 4).reshape(-1, n_action_steps, n_action_samples, action_dim)

        b_dppo_log_probs = dppo_log_probs_stored.reshape(-1, n_action_steps, actor_module.config.sampling_steps, num_envs_per_process)
        b_dppo_log_probs = b_dppo_log_probs.permute(0, 3, 1, 2).reshape(-1, n_action_steps, actor_module.config.sampling_steps)

        b_mdp_x_t_paths = mdp_x_t_paths_stored.reshape(-1, n_action_steps, actor_module.config.sampling_steps, num_envs_per_process, action_dim)
        b_mdp_x_t_paths = b_mdp_x_t_paths.permute(0, 3, 1, 2, 4).reshape(-1, n_action_steps, actor_module.config.sampling_steps, action_dim)

        b_cfm_value_invalid = cfm_value_invalid_stored.reshape(-1, n_action_steps, num_envs_per_process)
        b_cfm_value_invalid = b_cfm_value_invalid.permute(0, 2, 1).reshape(-1, n_action_steps)

        b_values = values_stored.reshape(-1, n_action_steps, num_envs_per_process)
        b_values = b_values.permute(0, 2, 1).reshape(-1, n_action_steps)

        b_dones = dones_stored.reshape(-1, n_action_steps, num_envs_per_process)
        b_dones = b_dones.permute(0, 2, 1).reshape(-1, n_action_steps)

        b_rewards = rewards_stored.reshape(-1, n_action_steps, num_envs_per_process)
        b_rewards = b_rewards.permute(0, 2, 1).reshape(-1, n_action_steps)

        b_obs_images = {
            k: obs_images_stored_list[k].reshape(-1, n_action_steps, num_envs_per_process, 3, img_h, img_w)
            for k in obs_images_stored_list.keys()
        }
        b_obs_images = {
            k: v.permute(0, 2, 1, 3, 4, 5).reshape(-1, n_action_steps, 3, img_h, img_w)
            for k, v in b_obs_images.items()
        }

        b_obs_state = obs_state_stored.reshape(-1, n_action_steps, num_envs_per_process, joint_pos_dim)
        b_obs_state = b_obs_state.permute(0, 2, 1, 3).reshape(-1, n_action_steps, joint_pos_dim)

        # ---------- Next value ----------
        if cfg.freeze_vision_encoder:
            actor_module.model.vision_encoder.eval()
        next_obs_cond = actor_module.normalize_inputs(next_obs)
        next_obs_cond = actor_module.model.encode_observations(next_obs_cond)
        next_value = critic(next_obs_cond).reshape(1, -1).cpu()

        # ---------- Advantages ----------
        advantages, returns = calculate_advantage(
            values_stored, next_value, rewards_stored, dones_stored, next_done,
            steps_per_iteration, cfg.discount, cfg.gae_lambda
        )
        b_advantages = advantages.reshape(-1, n_action_steps, num_envs_per_process).permute(0, 2, 1).reshape(-1, n_action_steps)
        b_returns = returns.reshape(-1, n_action_steps, num_envs_per_process).permute(0, 2, 1).reshape(-1, n_action_steps)

        # ---------- Policy update ----------
        b_inds = np.arange(local_batch_size)
        clipfracs = []
        actor.train(); critic.train()
        if cfg.freeze_vision_encoder:
            actor_module.model.vision_encoder.eval()

        # Initialize gradient norm tracking
        actor_grad_norm_before = torch.tensor(0.0)
        actor_grad_norm_after = torch.tensor(0.0)
        critic_grad_norm_before = torch.tensor(0.0)
        critic_grad_norm_after = torch.tensor(0.0)

        for epoch in trange(cfg.update_epochs, desc=f"[Rank {rank}] Policy update", disable=(rank != 0)):
            early_stop = False
            np.random.shuffle(b_inds)
            accumulation_counter = 0
            optimizer_actor.zero_grad(set_to_none=True)
            optimizer_critic.zero_grad(set_to_none=True)

            for start in range(0, local_batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_actions = b_actions[mb_inds].to(device)
                mb_cfm_losses = b_cfm_losses[mb_inds].to(device)
                mb_cfm_loss_ts = b_cfm_loss_ts[mb_inds].to(device)
                mb_cfm_loss_epsilons = b_cfm_loss_epsilons[mb_inds].to(device)
                mb_dppo_log_probs = b_dppo_log_probs[mb_inds].to(device)
                mb_mdp_x_t_paths = b_mdp_x_t_paths[mb_inds].to(device)
                mb_cfm_value_invalid = b_cfm_value_invalid[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)
                mb_obs_images = {k: b_obs_images[k][mb_inds].to(device) for k in b_obs_images.keys()}
                mb_obs_state = b_obs_state[mb_inds].to(device)

                valid_idx_mask_in_chunk = 1.0 - mb_cfm_value_invalid

                # Critic forward
                obs_chunk = {k: mb_obs_images[k].reshape(-1, 3, img_h, img_w) for k in mb_obs_images.keys()}
                obs_chunk["observation.state"] = mb_obs_state.reshape(-1, joint_pos_dim)
                obs_chunk = actor_module.normalize_inputs(obs_chunk)
                obs_chunk_cond = actor_module.model.encode_observations(obs_chunk)
                newvalue = critic(obs_chunk_cond)
                newvalue = newvalue.reshape(mb_returns.shape[0], -1)

                if cfg.loss_mode == "fpo":
                    # CFM losses
                    obs_chunk2 = {k: mb_obs_images[k][:, 0] for k in mb_obs_images.keys()}
                    obs_chunk2["observation.state"] = mb_obs_state[:, 0]
                    obs_chunk2["action"] = mb_actions

                    old_cfm_loss_ts = mb_cfm_loss_ts.permute(0, 2, 1).reshape(-1, n_action_steps, 1)
                    assert (old_cfm_loss_ts[:, 0, 0] == old_cfm_loss_ts[:, -1, 0]).all()
                    old_cfm_loss_ts = old_cfm_loss_ts[:, 0:1, :]
                    old_cfm_loss_epsilons = mb_cfm_loss_epsilons.permute(0, 2, 1, 3).reshape(-1, n_action_steps, action_dim)

                    curr_cfm_loss, _, _ = get_cfm_values(
                        actor, obs_chunk2, n_action_samples, old_cfm_loss_ts, old_cfm_loss_epsilons
                    )
                    curr_cfm_loss = curr_cfm_loss.permute(1, 0, 2)

                    old_cfm_loss = mb_cfm_losses.reshape(mb_cfm_losses.shape[0], -1, n_groups, group_size)
                    curr_cfm_loss = curr_cfm_loss.reshape(curr_cfm_loss.shape[0], -1, n_groups, group_size)

                    if cfg.clamp_old_cfm_loss is not None:
                        # old_cfm_loss = torch.clamp(old_cfm_loss, max=cfg.clamp_old_cfm_loss)
                        old_cfm_loss = clamp_ste(old_cfm_loss, max=cfg.clamp_old_cfm_loss)

                    if cfg.do_chunk_level_ppo:
                        if cfg.do_average_cfm_loss_in_chunk:
                            denom = valid_idx_mask_in_chunk.sum(dim=1).unsqueeze(-1).unsqueeze(-1).clamp_min(1.0)
                            old_cfm_loss = (old_cfm_loss * valid_idx_mask_in_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / denom
                            curr_cfm_loss = (curr_cfm_loss * valid_idx_mask_in_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / denom
                        else:
                            old_cfm_loss = (old_cfm_loss * valid_idx_mask_in_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
                            curr_cfm_loss = (curr_cfm_loss * valid_idx_mask_in_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

                        old_cfm_loss = old_cfm_loss.mean(dim=-1)
                        curr_cfm_loss = curr_cfm_loss.mean(dim=-1)
                        logratio = old_cfm_loss - curr_cfm_loss
                        if cfg.clamp_logratio is not None:
                            # logratio = torch.clamp(logratio, min=-cfg.clamp_logratio, max=cfg.clamp_logratio)
                            logratio = clamp_ste(logratio, min=-cfg.clamp_logratio, max=cfg.clamp_logratio)

                        ratio = logratio.exp()
                        mb_advantages = mb_advantages[:, 0:1]
                    else:
                        old_cfm_loss = old_cfm_loss.mean(dim=-1) # (B, T, n_groups)
                        curr_cfm_loss = curr_cfm_loss.mean(dim=-1) # (B, T, n_groups)
                        logratio = old_cfm_loss - curr_cfm_loss
                        if cfg.clamp_logratio is not None:
                            # logratio = torch.clamp(logratio, min=-cfg.clamp_logratio, max=cfg.clamp_logratio)
                            logratio = clamp_ste(logratio, min=-cfg.clamp_logratio, max=cfg.clamp_logratio)

                        ratio = logratio.exp()
                        mb_advantages = mb_advantages.unsqueeze(-1) # (B, T, 1)

                    # TODO
                    entropy = 0.0

                elif cfg.loss_mode == "dppo":
                    obs_chunk2 = {k: mb_obs_images[k][:, 0] for k in mb_obs_images.keys()}
                    obs_chunk2["observation.state"] = mb_obs_state[:, 0]
                    obs_chunk2["action"] = mb_actions

                    assert mb_mdp_x_t_paths.shape == (len(mb_inds), n_action_steps, actor_module.config.sampling_steps, action_dim)
                    obs_chunk2["mdp_x_t_path"] = mb_mdp_x_t_paths

                    log_prob, entropy, sde_sigma = get_log_prob_and_entropy(actor, obs_chunk2)
                    assert log_prob.shape == (len(mb_inds), n_action_steps, actor_module.config.sampling_steps)

                    # Get valid log probs
                    log_prob_valid = log_prob * valid_idx_mask_in_chunk.unsqueeze(-1)
                    mb_dppo_log_probs_valid = mb_dppo_log_probs * valid_idx_mask_in_chunk.unsqueeze(-1) 

                    # action chunk log prob
                    log_prob_chunk = log_prob_valid.sum(dim=1)
                    mb_dppo_log_probs_chunk = mb_dppo_log_probs_valid.sum(dim=1)                    

                    if cfg.average_logprob_over_denoising_steps:
                        # Average log probabilities over the flow steps
                        log_prob_chunk = log_prob_chunk.sum(dim=-1)
                        mb_dppo_log_probs_chunk = mb_dppo_log_probs_chunk.sum(dim=-1)

                        log_ratio = log_prob_chunk - mb_dppo_log_probs_chunk 
                        log_ratio = log_ratio.unsqueeze(-1)
                        assert log_ratio.shape == (len(mb_inds), 1), "log_ratio shape should be (len(mb_inds), 1)"

                        # log_ratio = log_ratio / (cfg.dppo_norm_factor * actor_module.config.sampling_steps * actor_module.config.horizon)
                        log_ratio = log_ratio / (cfg.dppo_norm_factor * actor_module.config.horizon)
                        ratio = log_ratio.exp()

                    else:
                        log_ratio = log_prob_chunk - mb_dppo_log_probs_chunk 

                        log_ratio = log_ratio / (cfg.dppo_norm_factor * actor_module.config.horizon)
                        ratio = log_ratio.exp()

                    # Do chunk level PPO
                    mb_advantages = mb_advantages[:, 0:1]

                else:
                    raise ValueError(f"Invalid loss mode: {cfg.loss_mode}")

                if cfg.norm_adv:
                    if is_ddp:
                        # Compute local statistics
                        adv_mean = mb_advantages.mean()
                        adv_std = mb_advantages.std()
                        # if adv_std is nan, set it to 0
                        if torch.isnan(adv_std):
                            adv_std = torch.tensor(0.0, device=device)
                        # Sync across ranks
                        stats = torch.tensor([adv_mean, adv_std, mb_advantages.numel()], device=device)
                        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                        global_mean = stats[0] / world_size
                        global_std = stats[1] / world_size
                        mb_advantages = (mb_advantages - global_mean) / (global_std + 1e-8)
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                if cfg.trust_region_mode == "spo":
                    clipfracs += [0.0]
                    spo_obj = mb_advantages * ratio - mb_advantages.abs() / (2.0 * cfg.spo_clip_coef) * (ratio - 1.0) ** 2
                    pg_loss = (-spo_obj).mean()
                elif cfg.trust_region_mode == "aspo":
                    clipfracs += [0.0]
                    spo_obj = mb_advantages * ratio - mb_advantages.abs() / (2.0 * cfg.spo_clip_coef) * (ratio - 1.0) ** 2
                    spo_obj = -spo_obj
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    # Keep the PPO surrogate PER-ELEMENT here; the outer torch.where(...).mean()
                    # below does the single reduction. Reducing to a scalar before the where
                    # (previous behavior) broadcast the batch-mean PPO gradient onto every
                    # positive-advantage sample and leaked it onto negatives too, so ASPO was
                    # optimizing a corrupted objective. Matches loco impl (flow_rsl_rl ppo.py).
                    ppo_obj = torch.max(pg_loss1, pg_loss2)
                    positive_mask = mb_advantages > 0
                    if cfg.do_chunk_level_ppo:
                        clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef)[positive_mask[:, 0], :].float().mean().item()]
                    else:
                        clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef)[positive_mask].float().mean().item()]
                    pg_loss = torch.where(mb_advantages > 0, ppo_obj, spo_obj).mean()
                else:
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values, -cfg.clip_coef, cfg.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2)
                v_loss = v_loss[valid_idx_mask_in_chunk == 1].mean()

                # Total loss
                entropy_loss = -entropy.mean() if cfg.learn_sde_sigma and isinstance(entropy, torch.Tensor) else 0.0
                policy_loss = pg_loss + cfg.entropy_loss_coef * entropy_loss if iteration > cfg.n_iterations_train_only_value else 0.0
                loss = (policy_loss + v_loss * cfg.vf_coef) / cfg.gradient_accumulation_steps
                loss.backward()
                accumulation_counter += 1

                if accumulation_counter % cfg.gradient_accumulation_steps == 0:
                    # Log gradient norms before clipping
                    actor_grad_norm_before = nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    critic_grad_norm_before = nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                    # Compute gradient norms after clipping
                    actor_grad_norm_after = torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'))
                    critic_grad_norm_after = torch.nn.utils.clip_grad_norm_(critic.parameters(), float('inf'))
                    optimizer_actor.step()
                    optimizer_critic.step()
                    if hasattr(actor_module, "step_ema"):
                        actor_module.step_ema()
                    optimizer_actor.zero_grad(set_to_none=True)
                    optimizer_critic.zero_grad(set_to_none=True)

            if accumulation_counter % cfg.gradient_accumulation_steps != 0:
                # Log gradient norms before clipping
                actor_grad_norm_before = nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                critic_grad_norm_before = nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                # Compute gradient norms after clipping
                actor_grad_norm_after = torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'))
                critic_grad_norm_after = torch.nn.utils.clip_grad_norm_(critic.parameters(), float('inf'))
                optimizer_actor.step()
                optimizer_critic.step()
                if hasattr(actor_module, "step_ema"):
                    actor_module.step_ema()
                optimizer_actor.zero_grad(set_to_none=True)
                optimizer_critic.zero_grad(set_to_none=True)

            if early_stop:
                break

        # Metrics (rank local)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        # Track CFM/DPPO metrics for logging (use last minibatch values as representative)
        if cfg.loss_mode == "fpo":
            cfm_metrics = {
                "cfm/old_cfm_loss_mean": float(old_cfm_loss.mean().item()),
                "cfm/old_cfm_loss_std": float(old_cfm_loss.std().item()),
                "cfm/curr_cfm_loss_mean": float(curr_cfm_loss.mean().item()),
                "cfm/curr_cfm_loss_std": float(curr_cfm_loss.std().item()),
                "cfm/logratio_mean": float(logratio.mean().item()),
                "cfm/logratio_std": float(logratio.std().item()),
                "cfm/logratio_min": float(logratio.min().item()),
                "cfm/logratio_max": float(logratio.max().item()),
                "cfm/ratio_mean": float(ratio.mean().item()),
                "cfm/ratio_std": float(ratio.std().item()),
                "cfm/ratio_min": float(ratio.min().item()),
                "cfm/ratio_max": float(ratio.max().item()),
                "cfm/old_cfm_loss_hist": wandb.Histogram(old_cfm_loss.detach().cpu().numpy().flatten()),
                "cfm/curr_cfm_loss_hist": wandb.Histogram(curr_cfm_loss.detach().cpu().numpy().flatten()),
            }
        else:  # dppo
            cfm_metrics = {
                "dppo/log_prob_chunk_mean": float(log_prob_chunk.mean().item()),
                "dppo/log_prob_chunk_std": float(log_prob_chunk.std().item()),
                "dppo/old_log_prob_chunk_mean": float(mb_dppo_log_probs_chunk.mean().item()),
                "dppo/old_log_prob_chunk_std": float(mb_dppo_log_probs_chunk.std().item()),
                "dppo/log_ratio_mean": float(log_ratio.mean().item()),
                "dppo/log_ratio_std": float(log_ratio.std().item()),
                "dppo/log_ratio_min": float(log_ratio.min().item()),
                "dppo/log_ratio_max": float(log_ratio.max().item()),
                "dppo/ratio_mean": float(ratio.mean().item()),
                "dppo/ratio_std": float(ratio.std().item()),
                "dppo/ratio_min": float(ratio.min().item()),
                "dppo/ratio_max": float(ratio.max().item()),
                "dppo/sde_sigma_mean": float(sde_sigma.mean().item()),
                "dppo/sde_sigma_std": float(sde_sigma.std().item()),
                "dppo/sde_sigma_min": float(sde_sigma.min().item()),
                "dppo/sde_sigma_max": float(sde_sigma.max().item()),
            }
            # Add entropy metrics (handles both learned and fixed sigma cases)
            if cfg.learn_sde_sigma and isinstance(entropy, torch.Tensor):
                # Learned sigma: entropy has shape (B, T)
                cfm_metrics["dppo/entropy_mean"] = float(entropy.mean().item())
                cfm_metrics["dppo/entropy_std"] = float(entropy.std().item())
            else:
                # Fixed sigma: entropy is scalar 0.0
                cfm_metrics["dppo/entropy_mean"] = float(entropy.item()) if isinstance(entropy, torch.Tensor) else float(entropy)

        training_cum_time += time.time() - iteration_start_time
        sps_local_cum = int((steps_per_iteration * num_envs_per_process) / (training_cum_time / max(1, iteration)))

        # Log (rank 0 only)
        if rank == 0 and iteration % cfg.log_freq == 0:
            msg = (
                f"[iter {iteration:>6d}/{num_iterations}]"
                f" SR: {success_rate_global:.2%}"
                f" | SPS_local_est: {sps_local_cum}"
                f" | pg_loss: {float(pg_loss):.4f}"
                f" | v_loss: {float(v_loss):.4f}"
                f" | total_loss: {float(loss):.4f}"
            )
            logger.info(colored(msg, "green"))
            if cfg.wandb_enable:
                log_dict = {
                    "training/learning_rate_actor": optimizer_actor.param_groups[0]["lr"],
                    "training/learning_rate_critic": optimizer_critic.param_groups[0]["lr"],
                    "training/SPS_local_est": sps_local_cum,
                    "training/actor_grad_norm_before_clip": float(actor_grad_norm_before),
                    "training/actor_grad_norm_after_clip": float(actor_grad_norm_after),
                    "training/critic_grad_norm_before_clip": float(critic_grad_norm_before),
                    "training/critic_grad_norm_after_clip": float(critic_grad_norm_after),
                    "charts/rewards": b_rewards.sum().item(),
                    "charts/success_rate": success_rate_global,
                    "charts/action_norm_mean": action_norms.mean(),
                    "charts/action_norm_std": action_norms.std(),
                    "values/advantages": b_advantages.mean().item(),
                    "values/returns": b_returns.mean().item(),
                    "values/values": b_values.mean().item(),
                    "losses/value_loss": float(v_loss),
                    "losses/policy_loss": float(pg_loss),
                    "losses/total_loss": float(loss),
                    "losses/clipfrac": float(np.mean(clipfracs)) if len(clipfracs) else 0.0,
                    "losses/explained_variance": explained_var,
                }
                # Add CFM/DPPO specific metrics
                log_dict.update(cfm_metrics)
                wandb.log(log_dict, step=global_step)

        # Step schedulers
        lr_scheduler_actor.step()
        lr_scheduler_critic.step()

        # Checkpointing (rank 0)
        if rank == 0 and cfg.save_freq > 0 and iteration % cfg.save_freq == 0:
            model_to_save = actor.module if is_ddp else actor
            ckpt_dir = (run_dir / "checkpoints") / f"step_{global_step}"
            save_checkpoint(ckpt_dir, global_step, model_to_save, optimizer_actor)

            latest_dir = (run_dir / "checkpoints") / "latest"
            if latest_dir.exists():
                import shutil
                shutil.rmtree(latest_dir)
            save_checkpoint(latest_dir, global_step, model_to_save, optimizer_actor)

            torch.save(
                {
                    "critic_state_dict": (critic.module.state_dict() if is_ddp else critic.state_dict()),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                    "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                    "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                    "config": vars(cfg),
                    "success_rate": success_rate_global,
                    "training_cum_time": training_cum_time,
                },
                latest_dir / "ppo_state.pt",
            )
            logger.info(colored(f"Checkpoint saved @ {ckpt_dir}", "magenta"))

            if cfg.wandb_enable and wandb.run is not None:
                try:
                    checkpoint_artifact = wandb.Artifact(
                        name=f"checkpoint_step_{global_step}",
                        type="model",
                        description=f"Model checkpoint at global_step {global_step}",
                        metadata={"step": global_step, "loss": float(loss)},
                    )
                    checkpoint_artifact.add_dir(str(ckpt_dir))
                    wandb.log_artifact(checkpoint_artifact)

                    latest_artifact = wandb.Artifact(
                        name="checkpoint_latest",
                        type="model",
                        description=f"Latest model checkpoint (global_step {global_step})",
                        metadata={"step": global_step, "loss": float(loss)},
                    )
                    latest_artifact.add_dir(str(latest_dir))
                    wandb.log_artifact(latest_artifact, aliases=["latest"])
                    logger.info(colored("Checkpoints uploaded to W&B", "magenta"))
                except Exception as e:
                    logger.warning(colored(f"Failed to upload checkpoint to W&B: {e}", "yellow"))

        # Evaluation (all ranks compute; rank 0 logs)
        if (
            cfg.rollout_freq is not None
            and cfg.eval_env is not None
            and (iteration % cfg.rollout_freq == 0 or iteration == num_iterations or iteration == 1)
        ):
            if is_ddp:
                dist.barrier()  # make sure training step is aligned

            t0 = time.perf_counter()
            results = eval_all_ranks(
                env=env,
                num_envs_per_process=num_envs_per_process,
                actor_ddp_or_single=actor,
                world_size=world_size,
                rank=rank,
                is_ddp=is_ddp,
                device_str=device_str,
                device=device,
                run_dir=run_dir,
                cfg=cfg,
                create_env_fn=create_vectorized_env,
                global_step=global_step,
            )
            if is_ddp:
                dist.barrier()

            if rank == 0:
                rollout_ms = (time.perf_counter() - t0) * 1000
                eval_sr = results["success_rate"]
                eval_fps = results["fps"]
                video_path = results["video_path"]
                episodes = results["episodes"]

                eval_sr_non_zero = results["success_rate_non_zero"]
                eval_fps_non_zero = results["fps_non_zero"]

                logger.info(colored(
                    f"[iteration {iteration:>6d}] global eval SR: {eval_sr*100:.1f}% | global episodes: {episodes} | "
                    f"rollout: {rollout_ms/1000:.2f}s | ~fps: {eval_fps:.1f}",
                    "cyan"
                ))

                if cfg.wandb_enable:
                    log_data = {
                        "eval/success_rate_zero_sampling": eval_sr,
                        "eval/fps_zero_sampling": eval_fps,
                        "eval/success_rate_random_sampling": eval_sr_non_zero,
                        "eval/fps_random_sampling": eval_fps_non_zero,
                        "eval/episodes_total": results["episodes"],
                        "time/rollout_ms": rollout_ms,
                    }
                    if video_path is not None and video_path.exists():
                        try:
                            video_artifact = wandb.Artifact(
                                name=f"eval_video_step_{global_step}",
                                type="video",
                                description=f"Evaluation video at global_step {global_step}",
                                metadata={"step": global_step, "success_rate": eval_sr},
                            )
                            video_artifact.add_file(str(video_path))
                            wandb.log_artifact(video_artifact)
                            log_data["eval/rollout_video"] = wandb.Video(str(video_path), format="mp4")
                        except Exception as e:
                            logger.warning(colored(f"Failed to upload video to W&B: {e}", "yellow"))
                    wandb.log(log_data, step=global_step)

            # Save best
            if rank == 0 and eval_sr > best_eval_success_rate:
                best_eval_success_rate = eval_sr
                logger.info(colored(f"New best success-rate! Saving checkpoint at global step {global_step}", "magenta"))
                model_to_save = actor.module if is_ddp else actor
                best_dir = (run_dir / "checkpoints") / "best"
                if best_dir.exists():
                    import shutil
                    shutil.rmtree(best_dir)
                save_checkpoint(best_dir, global_step, model_to_save, optimizer_actor)
                torch.save(
                    {
                        "critic_state_dict": (critic.module.state_dict() if is_ddp else critic.state_dict()),
                        "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                        "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                        "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                        "config": vars(cfg),
                        "success_rate": eval_sr,
                        "training_cum_time": training_cum_time,
                    },
                    best_dir / "ppo_state.pt",
                )
                if cfg.wandb_enable and wandb.run is not None:
                    try:
                        best_artifact = wandb.Artifact(
                            name="checkpoint_best",
                            type="model",
                            description=f"Best model checkpoint (global_step {global_step}, SR: {eval_sr * 100:.1f}%)",
                            metadata={"step": global_step, "success_rate": eval_sr},
                        )
                        best_artifact.add_dir(str(best_dir))
                        wandb.log_artifact(best_artifact, aliases=["best"])
                        logger.info(colored("Best checkpoint uploaded to W&B", "magenta"))
                    except Exception as e:
                        logger.warning(colored(f"Failed to upload best checkpoint to W&B: {e}", "yellow"))

        if is_ddp:
            dist.barrier()

    logger.info(colored("Training finished!", "green", attrs=["bold"]))
    if rank == 0:
        if cfg.wandb_enable:
            wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    args_cli = tyro.cli(FlowPPOConfig, config=(tyro.conf.FlagConversionOff,))
    main(args_cli)


