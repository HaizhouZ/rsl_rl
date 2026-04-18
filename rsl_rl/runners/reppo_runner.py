# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import time
import warnings

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import REPPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.modules import ActorQ
from rsl_rl.storage import ReppoRolloutStorage
from rsl_rl.utils import check_nan, resolve_obs_groups
from rsl_rl.utils.logger import Logger


class ReppoRunner:
    """Official-style REPPO runner with compatibility translation for legacy local configs."""

    alg: REPPO

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.device = device
        self.cfg = self._translate_train_cfg(copy.deepcopy(train_cfg))
        self.policy_cfg = self.cfg["policy"]
        self.alg_cfg = self.cfg["algorithm"]

        self._configure_multi_gpu()

        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())

        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        self.alg = self._construct_algorithm(obs)
        self.policy = self.alg.policy

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    if self.alg_cfg.get("scale_actions", False):
                        upper = self.alg_cfg.get("action_upper_bound", 1.0)
                        lower = self.alg_cfg.get("action_lower_bound", -1.0)
                        actions = actions * (upper - lower) / 2.0 + (upper + lower) / 2.0

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg.get("rnd_cfg") else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg.get("rnd_cfg") else None,
            )

            if self.gpu_global_rank == 0 and self.logger.log_dir is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.gpu_global_rank == 0 and self.logger.log_dir is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg_cfg.get("rnd_cfg"):
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer is not None:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()

        torch.save(saved_dict, path)
        if getattr(self.logger, "writer", None) is not None:
            self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if self.alg_cfg.get("rnd_cfg"):
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg_cfg.get("rnd_cfg") and self.alg.rnd_optimizer is not None:
                rnd_optimizer_state = loaded_dict.get("rnd_optimizer_state_dict")
                if rnd_optimizer_state is not None:
                    self.alg.rnd_optimizer.load_state_dict(rnd_optimizer_state)

        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        self.alg.policy.train()
        if self.alg_cfg.get("rnd_cfg"):
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()
        if self.alg_cfg.get("rnd_cfg"):
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _construct_algorithm(self, obs: TensorDict) -> REPPO:
        policy = ActorQ(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(self.device)
        storage = ReppoRolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )
        return REPPO(policy, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

    def _get_default_obs_sets(self) -> list[str]:
        default_sets = ["policy", "critic"]
        if self.alg_cfg.get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        return default_sets

    @staticmethod
    def _translate_train_cfg(train_cfg: dict) -> dict:
        cfg = copy.deepcopy(train_cfg)
        cfg["obs_groups"] = ReppoRunner._translate_obs_groups(cfg.get("obs_groups", {}))

        policy_cfg = dict(cfg.get("policy", {}))
        algorithm_cfg = dict(cfg.get("algorithm", {}))

        actor_hidden_dims = ReppoRunner._resolve_actor_hidden_dims(policy_cfg)
        critic_hidden_dims = ReppoRunner._resolve_critic_hidden_dims(policy_cfg)

        translated_policy = {
            "actor_obs_normalization": policy_cfg.pop("actor_obs_normalization", False),
            "critic_obs_normalization": policy_cfg.pop("critic_obs_normalization", False),
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "num_critic_bins": algorithm_cfg.pop("num_atoms", 151),
            "vmin": algorithm_cfg.pop("vmin", -10.0),
            "vmax": algorithm_cfg.pop("vmax", 10.0),
            "activation": policy_cfg.pop("activation", "elu"),
            "init_noise_std": policy_cfg.pop("init_noise_std", 1.0),
            "noise_std_type": policy_cfg.pop("noise_std_type", "scalar"),
            "state_dependent_std": policy_cfg.pop("state_dependent_std", True),
            "distribution_type": policy_cfg.pop("distribution_type", "tanh"),
            "init_alpha_temp": policy_cfg.pop("ent_start", 0.001),
            "init_alpha_kl": policy_cfg.pop("kl_start", 0.01),
            "action_lower_bound": policy_cfg.pop("action_lower_bound", -1.0),
            "action_upper_bound": policy_cfg.pop("action_upper_bound", 1.0),
        }

        policy_cfg.pop("class_name", None)
        policy_cfg.pop("critic_class_name", None)
        ignored_policy = {
            key: policy_cfg.pop(key)
            for key in list(policy_cfg)
            if key
            in {
                "actor_min_std",
                "reset_global_std_on_resume",
                "use_actor_norm",
                "use_critic_norm",
                "use_encoder_norm",
                "num_critic_pred_layers",
            }
        }
        ReppoRunner._warn_ignored_fields("policy", ignored_policy)
        ReppoRunner._warn_ignored_fields("policy", policy_cfg)

        target_entropy = algorithm_cfg.pop("target_entropy", None)
        if target_entropy is None:
            target_entropy = -abs(float(algorithm_cfg.pop("ent_target_mult", 0.5)))
        else:
            target_entropy = float(target_entropy)

        translated_algorithm = {
            "num_learning_epochs": algorithm_cfg.pop("num_learning_epochs", 4),
            "num_mini_batches": algorithm_cfg.pop("num_mini_batches", 4),
            "gamma": algorithm_cfg.pop("gamma", 0.99),
            "lam": algorithm_cfg.pop("lam", algorithm_cfg.pop("lmbda", 0.95)),
            "learning_rate": algorithm_cfg.pop("learning_rate", 3e-4),
            "max_grad_norm": algorithm_cfg.pop("max_grad_norm", 0.5),
            "desired_kl": algorithm_cfg.pop("kl_bound", algorithm_cfg.pop("desired_kl", 0.01)),
            "target_entropy": target_entropy,
            "rnd_cfg": algorithm_cfg.pop("rnd_cfg", None),
            "symmetry_cfg": algorithm_cfg.pop("symmetry_cfg", None),
            "scale_actions": algorithm_cfg.pop("scale_actions", False),
            "action_lower_bound": translated_policy["action_lower_bound"],
            "action_upper_bound": translated_policy["action_upper_bound"],
        }

        algorithm_cfg.pop("class_name", None)
        ignored_algorithm = {
            key: algorithm_cfg.pop(key)
            for key in list(algorithm_cfg)
            if key
            in {
                "aux_loss_mult",
                "actor_kl_clip_mode",
                "schedule",
                "optimizer",
                "entropy_coef",
                "value_loss_coef",
                "use_clipped_value_loss",
                "clip_param",
                "normalize_advantage_per_mini_batch",
            }
        }
        ReppoRunner._warn_ignored_fields("algorithm", ignored_algorithm)
        ReppoRunner._warn_ignored_fields("algorithm", algorithm_cfg)

        cfg["policy"] = translated_policy
        cfg["algorithm"] = translated_algorithm
        return cfg

    @staticmethod
    def _translate_obs_groups(obs_groups: dict[str, list[str] | tuple[str, ...]]) -> dict[str, list[str] | tuple[str, ...]]:
        translated = copy.deepcopy(obs_groups)
        if "policy" not in translated and "actor" in translated:
            translated["policy"] = translated.pop("actor")
        return translated

    @staticmethod
    def _resolve_actor_hidden_dims(policy_cfg: dict) -> tuple[int, ...]:
        actor_hidden_dims = tuple(policy_cfg.pop("actor_hidden_dims", ()))
        actor_hidden_dim = int(policy_cfg.pop("actor_hidden_dim", 512))
        num_actor_layers = int(policy_cfg.pop("num_actor_layers", 3))
        if actor_hidden_dims:
            return actor_hidden_dims
        return tuple(actor_hidden_dim for _ in range(max(num_actor_layers - 1, 1)))

    @staticmethod
    def _resolve_critic_hidden_dims(policy_cfg: dict) -> tuple[int, ...]:
        critic_hidden_dims = tuple(policy_cfg.pop("critic_hidden_dims", ()))
        critic_hidden_dim = int(policy_cfg.pop("critic_hidden_dim", 512))
        num_critic_encoder_layers = int(policy_cfg.pop("num_critic_encoder_layers", 2))
        num_critic_head_layers = int(policy_cfg.pop("num_critic_head_layers", 2))
        total_hidden_layers = max(num_critic_encoder_layers + num_critic_head_layers - 1, 1)
        if critic_hidden_dims:
            if len(critic_hidden_dims) == 1:
                return (critic_hidden_dims[0], critic_hidden_dims[0])
            return critic_hidden_dims
        return tuple(critic_hidden_dim for _ in range(total_hidden_layers))

    @staticmethod
    def _warn_ignored_fields(scope: str, ignored_cfg: dict) -> None:
        if ignored_cfg:
            warnings.warn(
                f"REPPO {scope} options are ignored by the official ActorQ/REPPO port: {sorted(ignored_cfg)}",
                stacklevel=3,
            )

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)
