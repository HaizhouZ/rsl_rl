# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.modules import ActorQ
from rsl_rl.storage.reppo_rollout_storage import ReppoRolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class REPPO:
    """Official feedforward REPPO algorithm."""

    policy: ActorQ

    def __init__(
        self,
        policy: ActorQ,
        storage: ReppoRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        desired_kl: float = 0.01,
        target_entropy: float = -1.0,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        if rnd_cfg:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            if getattr(policy, "is_recurrent", False):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        self.policy = policy.to(self.device)
        self.old_policy = copy.deepcopy(self.policy).to(self.device)
        self.old_policy.eval()
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=learning_rate, weight_decay=1e-3, betas=(0.9, 0.95))
        self.storage = storage
        self.transition = ReppoRolloutStorage.Transition()

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.target_entropy = target_entropy * self.policy.num_actions
        self.learning_rate = learning_rate

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs, self.transition.actions).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool)).to(self.device)
        time_outs_bool = time_outs.bool()
        time_outs_float = time_outs.float()
        dones_bool = dones.bool()

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones_bool & ~time_outs_bool
        self.transition.truncations = time_outs_float

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * self.transition.values * time_outs_float

        self.transition.soft_rewards = (
            self.transition.rewards - self.gamma * self.policy.alpha_temp * self.transition.actions_log_prob
        )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones_bool)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        last_action = self.policy.act(obs).detach()
        last_values = self.policy.evaluate(obs, last_action).detach().view(-1, 1)
        recurr_value = last_values
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta_1 = next_is_not_terminal * self.gamma * next_values
            delta_n = next_is_not_terminal * self.gamma * recurr_value
            recurr_value = st.soft_rewards[step] + (1 - self.lam) * delta_1 + self.lam * delta_n
            st.returns[step] = recurr_value

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        with torch.no_grad():
            self.old_policy.load_state_dict(self.policy.state_dict())

        generator = self.storage.recurrent_mini_batch_generator if self.policy.is_recurrent else self.storage.mini_batch_generator
        for (
            obs_batch,
            actions_batch,
            _,
            _,
            returns_batch,
            truncations_batch,
            _,
            old_mean_batch,
            old_std_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator(self.num_mini_batches, self.num_learning_epochs):
            critic_metrics = self.update_critic(
                {
                    "obs_batch": obs_batch,
                    "actions_batch": actions_batch,
                    "returns_batch": returns_batch,
                    "truncations_batch": truncations_batch,
                    "hidden_states_batch": hidden_states_batch,
                    "masks_batch": masks_batch,
                    "old_mean": old_mean_batch,
                    "old_std": old_std_batch,
                }
            )
            mean_value_loss += critic_metrics["value_loss"]

        for (
            obs_batch,
            actions_batch,
            _,
            _,
            returns_batch,
            truncations_batch,
            _,
            old_mean_batch,
            old_std_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator(self.num_mini_batches, self.num_learning_epochs):
            actor_metrics = self.update_actor(
                {
                    "obs_batch": obs_batch,
                    "actions_batch": actions_batch,
                    "returns_batch": returns_batch,
                    "truncations_batch": truncations_batch,
                    "hidden_states_batch": hidden_states_batch,
                    "masks_batch": masks_batch,
                    "old_mean": old_mean_batch,
                    "old_std": old_std_batch,
                }
            )
            mean_entropy += actor_metrics["entropy"]
            mean_surrogate_loss += actor_metrics["actor_loss"]

        print("value prediction error: ", critic_metrics["value_prediction_error"])
        print("enc dec error: ", critic_metrics["enc_dec_error"])
        print("on policy values mean: ", actor_metrics["on_policy_values_mean"])
        print("entropy: ", actor_metrics["entropy"])
        print("kl divergence: ", actor_metrics["kl_divergence"])
        print("entropy target: ", self.target_entropy)
        print("alpha temp: ", self.policy.alpha_temp.item())
        print("alpha kl: ", self.policy.alpha_kl.item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def update_actor(self, minibatch: dict) -> dict:
        obs_batch = minibatch["obs_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        self.policy.act(obs_batch, hidden_states_batch, masks_batch)
        predicted_policy = self.policy.distribution
        predicted_actions = predicted_policy.rsample()
        on_policy_values = self.policy.evaluate(obs_batch, predicted_actions, hidden_states_batch, masks_batch)

        entropy = -predicted_policy.log_prob(predicted_actions).sum(-1)
        entropy_loss = self.policy.alpha_temp.detach() * entropy
        primary_policy_loss = -(on_policy_values + entropy_loss)

        with torch.no_grad():
            self.old_policy.act(obs_batch, hidden_states_batch, masks_batch)
            old_policy_distribution = self.old_policy.distribution
            old_policy_actions = old_policy_distribution.sample((4,))
            log_prob_old = old_policy_distribution.log_prob(old_policy_actions).detach()
        log_prob_new = predicted_policy.log_prob(old_policy_actions)
        kl_divergence = (log_prob_old - log_prob_new).sum(-1).mean(0)

        policy_loss = torch.where(
            (kl_divergence < self.desired_kl).detach(),
            primary_policy_loss,
            self.policy.alpha_kl.detach() * kl_divergence,
        ).mean()

        temp_target_loss = self.policy.alpha_temp * (entropy.mean() - self.target_entropy).detach()
        kl_target_loss = self.policy.alpha_kl * (self.desired_kl - kl_divergence.mean()).detach()

        self._set_critic_grad(False)
        self.optimizer.zero_grad()
        actor_loss = policy_loss + temp_target_loss + kl_target_loss
        actor_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._set_critic_grad(True)

        return {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "kl_divergence": kl_divergence.mean().item(),
            "on_policy_values_mean": on_policy_values.mean().item(),
        }

    def update_critic(self, minibatch: dict) -> dict:
        obs_batch = minibatch["obs_batch"]
        actions_batch = minibatch["actions_batch"]
        returns_batch = minibatch["returns_batch"]
        truncations_batch = minibatch["truncations_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        value_prediction, value_logits = self.policy.evaluate(
            obs_batch, actions_batch, hidden_states_batch, masks_batch, return_logits=True
        )
        embedded_returns = self.policy.hlgauss_embed(returns_batch.view(-1)).view(value_logits.shape).detach()
        value_loss_per_sample = -(embedded_returns * torch.log_softmax(value_logits, dim=-1)).sum(-1)
        value_loss_weights = 1.0 - truncations_batch.view(-1).float()
        value_loss = (value_loss_weights * value_loss_per_sample).sum() / value_loss_weights.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        value_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        value_prediction_error = (value_prediction.view(-1) - returns_batch.view(-1)).abs().mean().item()
        decoded_returns = self.policy.hlgauss_decode(torch.log(embedded_returns)).detach()
        enc_dec_error = (decoded_returns.view(-1) - returns_batch.view(-1)).abs().mean().item()
        return {
            "value_loss": value_loss.item(),
            "value_prediction_error": value_prediction_error,
            "enc_dec_error": enc_dec_error,
        }

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def _set_critic_grad(self, requires_grad: bool) -> None:
        for param in self.policy.critic.parameters():
            param.requires_grad = requires_grad
        for param in self.policy.critic_embedding_layer.parameters():
            param.requires_grad = requires_grad
        for param in self.policy.norm.parameters():
            param.requires_grad = requires_grad

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.policy.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.policy.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None:
            load_cfg = {
                "policy": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        if load_cfg.get("policy"):
            self.policy.load_state_dict(loaded_dict["policy_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> ActorQ:
        """Get the policy model."""
        return self.policy

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> REPPO:
        """Construct the REPPO algorithm for :class:`OnPolicyRunner`."""
        alg_class: type[REPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
        policy_class: type[ActorQ] = resolve_callable(cfg["policy"].pop("class_name"))  # type: ignore[assignment]

        default_sets = ["policy", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        policy = policy_class(obs, cfg["obs_groups"], env.num_actions, **cfg["policy"]).to(device)
        storage = ReppoRolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        return alg_class(policy, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
