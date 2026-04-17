# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import time
from collections.abc import Iterable
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import ReppoCritic, ReppoPolicy
from rsl_rl.models.reppo_model import hl_gauss
from rsl_rl.utils import check_nan, resolve_callable, resolve_obs_groups, resolve_optimizer
from rsl_rl.utils.logger import Logger


class ReppoRunner:
    """Runner for REPPO training on vectorized environments."""

    policy: ReppoPolicy
    critic: ReppoCritic

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device

        self._configure_multi_gpu()

        obs = self.env.get_observations()

        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], ["actor", "critic"])

        policy_cfg = dict(self.cfg["policy"])
        algorithm_cfg = self.cfg["algorithm"]
        self.num_learning_epochs = algorithm_cfg.get("num_learning_epochs", 4)
        self.num_mini_batches = algorithm_cfg.get("num_mini_batches", 32)
        self.learning_rate = algorithm_cfg.get("learning_rate", 3e-4)
        self.gamma = algorithm_cfg.get("gamma", 0.99)
        self.lmbda = algorithm_cfg.get("lmbda", algorithm_cfg.get("lam", 0.95))
        self.num_atoms = algorithm_cfg.get("num_atoms", 151)
        self.vmin = algorithm_cfg.get("vmin", 0.0)
        self.vmax = algorithm_cfg.get("vmax", 150.0)
        self.aux_loss_mult = algorithm_cfg.get("aux_loss_mult", 0.0)
        self.num_action_samples = algorithm_cfg.get("num_action_samples", 64)
        self.num_action_sample_chunk_size = algorithm_cfg.get("num_action_sample_chunk_size", 1)
        self.kl_bound = algorithm_cfg.get("kl_bound", algorithm_cfg.get("desired_kl", 0.1))
        self.actor_kl_clip_mode = algorithm_cfg.get("actor_kl_clip_mode", "clipped")
        self.ent_target_mult = algorithm_cfg.get("ent_target_mult", 0.5)
        policy_class = resolve_callable(policy_cfg.pop("class_name", "ReppoPolicy"))  # type: ignore
        critic_class = resolve_callable(policy_cfg.pop("critic_class_name", "ReppoCritic"))  # type: ignore

        self.policy = policy_class(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(self.device)
        self.critic = critic_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
            num_atoms=self.num_atoms,
            vmin=self.vmin,
            vmax=self.vmax,
            aux_loss_mult=self.aux_loss_mult,
        ).to(self.device)
        self.policy_old = copy.deepcopy(self.policy).to(self.device)
        self.policy_old.eval()

        optimizer_name = algorithm_cfg.get("optimizer", "adamw")
        self.actor_optimizer = resolve_optimizer(optimizer_name)(self.policy.parameters(), lr=self.learning_rate)
        self.critic_optimizer = resolve_optimizer(optimizer_name)(
            self.critic.parameters(), lr=self.learning_rate
        )

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
        self.pending_transition: dict[str, torch.Tensor] | None = None
        self.transitions: list[dict[str, torch.Tensor]] = []
        self.rollout_data: dict[str, torch.Tensor] | None = None
        self.alg = SimpleNamespace(policy=self.policy, critic=self.critic)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            self.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            for _ in range(self.cfg["num_steps_per_env"]):
                actions = self.act(obs)
                obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                if self.cfg.get("check_for_nan", True):
                    check_nan(obs, rewards, dones)
                obs = obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                self.process_env_step(obs, rewards, dones, extras)
                self.logger.process_env_step(rewards, dones, extras)

            stop = time.time()
            collect_time = stop - start
            start = stop

            self.compute_returns(obs)
            loss_dict = self.update()

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
                learning_rate=self.actor_optimizer.param_groups[0]["lr"],
                action_std=self.policy.output_std,
                rnd_weight=None,
            )

            if self.gpu_global_rank == 0 and self.logger.log_dir is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.gpu_global_rank == 0 and self.logger.log_dir is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
        self.logger.stop_logging_writer()

    def act(self, obs: TensorDict) -> torch.Tensor:
        raw_actor_obs = self.policy.get_actor_obs(obs)
        raw_critic_obs = self.critic.get_critic_obs(obs)
        actor_obs = self.policy.normalize_actor_obs(raw_actor_obs)
        actions, log_prob, entropy, _, _, _ = self.policy.sample_actions_from_normalized(actor_obs)
        self.pending_transition = {
            "observations": raw_actor_obs.detach(),
            "critic_observations": raw_critic_obs.detach(),
            "actions": actions.detach(),
            "log_probs": log_prob.detach(),
            "entropy": entropy.detach(),
        }
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self.pending_transition is None:
            raise RuntimeError("process_env_step called before act")

        effective_next_obs = obs
        if (
            self.cfg.get("env", {}).get("has_final_obs", False)
            and self.cfg.get("env", {}).get("partial_reset", False)
            and "final_observation" in extras
        ):
            maybe_final_obs = extras["final_observation"]
            if isinstance(maybe_final_obs, TensorDict):
                effective_next_obs = maybe_final_obs

        next_actor_obs = self.policy.get_actor_obs(effective_next_obs)
        next_critic_obs = self.critic.get_critic_obs(effective_next_obs)

        truncations = extras.get("time_outs")
        if truncations is None:
            truncations = torch.zeros_like(dones)

        rewards = rewards.reshape(rewards.shape[0], -1)
        dones = dones.reshape(dones.shape[0], -1)
        truncations = truncations.reshape(truncations.shape[0], -1)

        transition = dict(self.pending_transition)
        transition.update(
            {
                "rewards": rewards.clone().to(dtype=torch.float32),
                "dones": dones.to(dtype=torch.float32),
                "truncations": truncations.to(dtype=torch.float32),
                "next_observations": next_actor_obs.detach(),
                "next_critic_observations": next_critic_obs.detach(),
            }
        )
        self.transitions.append(transition)
        self.pending_transition = None

    def compute_returns(self, obs: TensorDict) -> None:
        if not self.transitions:
            self.rollout_data = None
            return

        data = self._stack_transitions(self.transitions)
        data = self._normalize_rollout(data)
        rollout_extras = self._compute_rollout_extras(data)
        data.update(rollout_extras)
        data["gve"] = self._compute_gve(
            rewards=data["rewards"],
            dones=data["dones"],
            truncations=data["truncations"],
            next_values=data["next_values"],
        )
        self.rollout_data = data

    def update(self) -> dict[str, float]:
        if self.rollout_data is None:
            return {}

        data = self.rollout_data
        obs = data["observations"].flatten(0, 1)
        critic_obs = data["critic_observations"].flatten(0, 1)
        actions = data["actions"].flatten(0, 1)
        rewards = data["rewards"].flatten(0, 1)
        dones = data["dones"].flatten(0, 1)
        truncations = data["truncations"].flatten(0, 1)
        next_values = data["next_values"].flatten(0, 1)
        next_embeddings = data["next_embeddings"].flatten(0, 1)
        gve = data["gve"].flatten(0, 1)

        batch_size = max(1, obs.shape[0] // self.num_mini_batches)
        indices = torch.arange(obs.shape[0], device=self.device)

        partial_reset = bool(self.cfg.get("env", {}).get("partial_reset", False))
        truncation_mask = torch.ones_like(truncations) if partial_reset else 1.0 - truncations

        mean_critic_loss = 0.0
        mean_actor_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        mean_embedding_loss = 0.0
        num_updates = 0

        old_policy = self.policy_old

        for _ in range(self.num_learning_epochs):
            perm = indices[torch.randperm(indices.numel(), device=self.device)]
            for start in range(0, perm.numel(), batch_size):
                batch_idx = perm[start : start + batch_size]

                batch_obs = obs[batch_idx]
                batch_critic_obs = critic_obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_gve = gve[batch_idx]
                batch_next_embeddings = next_embeddings[batch_idx]
                batch_mask = truncation_mask[batch_idx]

                qf_target_dist = hl_gauss(
                    batch_gve,
                    self.vmin,
                    self.vmax,
                    self.num_atoms,
                )

                _, logits, _, embedding = self.critic.forward_normalized(batch_critic_obs, batch_actions)
                qf_loss = -(
                    batch_mask * torch.sum(qf_target_dist * F.log_softmax(logits, dim=-1), dim=-1)
                ).mean()
                embedding_loss = (
                    batch_mask
                    * F.mse_loss(embedding, batch_next_embeddings, reduction="none")
                ).mean()
                critic_loss = qf_loss + self.aux_loss_mult * embedding_loss

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                if self.is_distributed:
                    self.reduce_parameters(self.critic.parameters())
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg["algorithm"].get("max_grad_norm", 1.0)
                )
                self.critic_optimizer.step()

                # Actor update
                (
                    new_actions,
                    log_probs,
                    entropy,
                    _,
                    temperature,
                    beta,
                ) = self.policy.sample_actions_from_normalized(batch_obs)
                qf, _, _, _ = self.critic.forward_normalized(batch_critic_obs, new_actions)

                with torch.inference_mode():
                    old_dist = old_policy.build_distribution_from_normalized(batch_obs)
                    new_dist = self.policy.build_distribution_from_normalized(batch_obs)
                    old_actions = old_dist.sample((16,)).clamp(-1 + 1e-6, 1 - 1e-6)
                    old_log_probs = old_dist.log_prob(old_actions).sum(dim=-1).mean(dim=0)
                    new_log_probs = new_dist.log_prob(old_actions).sum(dim=-1).mean(dim=0)
                    kl = old_log_probs - new_log_probs

                actor_loss = -qf + temperature.detach() * log_probs

                if self.actor_kl_clip_mode == "clipped":
                    actor_loss = torch.where(kl < self.kl_bound, actor_loss, kl * beta.detach())
                elif self.actor_kl_clip_mode == "full":
                    actor_loss = actor_loss + kl * beta.detach()
                elif self.actor_kl_clip_mode == "value":
                    actor_loss = actor_loss
                else:
                    raise ValueError(f"Unknown actor_kl_clip_mode: {self.actor_kl_clip_mode}")

                target_entropy = new_actions.shape[-1] * self.ent_target_mult
                entropy_loss = (target_entropy + entropy).detach().mean() * temperature
                lagrangian_loss = (-beta * (kl - self.kl_bound).mean().detach())
                actor_loss = (actor_loss + entropy_loss + lagrangian_loss).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.is_distributed:
                    self.reduce_parameters(self.policy.parameters())
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg["algorithm"].get("max_grad_norm", 1.0)
                )
                self.actor_optimizer.step()

                mean_critic_loss += critic_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_entropy += entropy.mean().item()
                mean_kl += kl.mean().item()
                mean_embedding_loss += embedding_loss.item()
                num_updates += 1

        if num_updates == 0:
            raise RuntimeError("REPPO update produced zero optimization batches")

        mean_critic_loss /= num_updates
        mean_actor_loss /= num_updates
        mean_entropy /= num_updates
        mean_kl /= num_updates
        mean_embedding_loss /= num_updates

        self.transitions.clear()
        self.rollout_data = None
        self.policy_old.load_state_dict(self.policy.state_dict())

        return {
            "critic": mean_critic_loss,
            "actor": mean_actor_loss,
            "entropy": mean_entropy,
            "kl": mean_kl,
            "embedding": mean_embedding_loss,
        }

    def train_mode(self) -> None:
        self.policy.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.policy.eval()
        self.critic.eval()

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = self._build_save_dict(infos=infos)
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def _build_save_dict(self, infos: dict | None = None) -> dict:
        env_state = {}
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "common_step_counter"):
            env_state["common_step_counter"] = self.env.unwrapped.common_step_counter
        return {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "policy_old_state_dict": self.policy_old.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "env_state": env_state,
        }

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        self.policy.load_state_dict(loaded_dict["policy_state_dict"], strict=strict)
        self.policy_old.load_state_dict(loaded_dict["policy_old_state_dict"], strict=strict)
        self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_optimizer:
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        if "env_state" in loaded_dict and hasattr(self.env, "unwrapped"):
            env_state = loaded_dict["env_state"] or {}
            if "common_step_counter" in env_state and hasattr(self.env.unwrapped, "common_step_counter"):
                self.env.unwrapped.common_step_counter = env_state["common_step_counter"]
        return loaded_dict.get("infos") or {}

    def get_inference_policy(self, device: str | None = None) -> ReppoPolicy:
        self.policy.eval()
        return self.policy.to(device) if device is not None else self.policy

    def get_policy(self) -> ReppoPolicy:
        return self.policy

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict(), self.critic.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.policy_old.load_state_dict(model_params[0])

    def reduce_parameters(self, params: Iterable[torch.nn.Parameter] | None = None) -> None:
        all_params = list(params) if params is not None else list(self.policy.parameters()) + list(self.critic.parameters())
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.cfg["multi_gpu"] = {
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

    def _stack_transitions(self, transitions: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = transitions[0].keys()
        stacked: dict[str, torch.Tensor] = {}
        for key in keys:
            stacked[key] = torch.stack([transition[key] for transition in transitions], dim=0)
        return stacked

    def _compute_gve(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncations: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        gves = []
        last_gve = torch.zeros_like(next_values[0])
        truncations = truncations.clone()
        truncations[-1] = 1.0
        for t in reversed(range(self.cfg["num_steps_per_env"])):
            lambda_sum = self.lmbda * last_gve + (1.0 - self.lmbda) * next_values[t]
            delta = self.gamma * torch.where(
                truncations[t].bool(), next_values[t], (1.0 - dones[t]) * lambda_sum
            )
            last_gve = rewards[t] + delta
            gves.insert(0, last_gve)
        return torch.stack(gves)

    def _compute_rollout_extras(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        next_obs = data["next_observations"]
        next_critic_obs = data["next_critic_observations"]
        rewards = data["rewards"]
        actions = data["actions"]
        truncations = data["truncations"]

        with torch.inference_mode():
            next_dist = self.policy.build_distribution_from_normalized(next_obs)
            sampled_next_actions = next_dist.sample()
            shifted_actions = torch.cat([actions[1:], sampled_next_actions[-1:]], dim=0)
            true_next_actions = torch.where(
                truncations.bool().expand_as(sampled_next_actions),
                sampled_next_actions,
                shifted_actions,
            ).clamp(-1 + 1e-6, 1 - 1e-6)
            next_log_probs = next_dist.log_prob(true_next_actions).sum(dim=-1, keepdim=True)
            temperature = torch.exp(self.policy.log_temp)
            soft_rewards = rewards - self.gamma * next_log_probs * temperature

            value_sum = None
            next_embeddings = None
            remaining_samples = self.num_action_samples
            while remaining_samples > 0:
                chunk_size = min(self.num_action_sample_chunk_size, remaining_samples)
                sampled_value_actions = next_dist.sample(sample_shape=(chunk_size,))
                expanded_next_critic_obs = next_critic_obs.unsqueeze(0).expand(chunk_size, *next_critic_obs.shape)
                next_values, _, _, chunk_embeddings = self.critic.forward_normalized(
                    expanded_next_critic_obs, sampled_value_actions
                )
                chunk_value_sum = next_values.sum(dim=0)
                value_sum = chunk_value_sum if value_sum is None else value_sum + chunk_value_sum
                if next_embeddings is None:
                    next_embeddings = chunk_embeddings[0]
                remaining_samples -= chunk_size

        if value_sum is None or next_embeddings is None:
            raise RuntimeError("REPPO next-value estimation produced no samples")

        return {
            "rewards": soft_rewards.to(dtype=torch.float32),
            "next_values": (value_sum / float(self.num_action_samples)).unsqueeze(-1).to(dtype=torch.float32),
            "next_embeddings": next_embeddings.to(dtype=torch.float32),
        }

    def _normalize_rollout(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        observations = data["observations"]
        critic_observations = data["critic_observations"]
        next_observations = data["next_observations"]
        next_critic_observations = data["next_critic_observations"]

        normalized_data = dict(data)
        normalized_data["observations"] = self.policy.normalize_actor_obs(observations).detach()
        normalized_data["critic_observations"] = self.critic.normalize_critic_obs(critic_observations).detach()
        normalized_data["next_observations"] = self.policy.normalize_actor_obs(next_observations).detach()
        normalized_data["next_critic_observations"] = self.critic.normalize_critic_obs(next_critic_observations).detach()

        if self.policy.actor_obs_normalization:
            self.policy.actor_obs_normalizer.update(observations.flatten(0, -2))  # type: ignore[operator]
        if self.critic.critic_obs_normalization:
            self.critic.critic_obs_normalizer.update(critic_observations.flatten(0, -2))  # type: ignore[operator]

        return normalized_data
