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
        clip_param: float = 0.2,
        actor_route: str = "reppo",
        ppo_advantage_normalization: bool = False,
        ppo_entropy_coef: float = 0.01,
        ppo_value_loss_coef: float = 1.0,
        ppo_schedule: str = "adaptive",
        cosine_weight_min: float = 0.0,
        cosine_weight_max: float = 1.0,
        cosine_weight_power: float = 1.0,
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

        valid_actor_routes = {"reppo", "hybrid", "ppo_only"}
        if actor_route not in valid_actor_routes:
            raise ValueError(
                f"Unsupported REPPO actor_route={actor_route!r}. Expected one of {valid_actor_routes}."
            )

        self.policy = policy.to(self.device)
        self.old_policy = copy.deepcopy(self.policy).to(self.device)
        self.old_policy.eval()
        if actor_route in {"hybrid", "ppo_only"}:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        else:
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
        self.clip_param = clip_param
        self.actor_route = actor_route
        self.ppo_advantage_normalization = ppo_advantage_normalization
        self.ppo_entropy_coef = ppo_entropy_coef
        self.ppo_value_loss_coef = ppo_value_loss_coef
        self.ppo_schedule = ppo_schedule
        self.cosine_weight_min = cosine_weight_min
        self.cosine_weight_max = cosine_weight_max
        self.cosine_weight_power = cosine_weight_power

        if self.clip_param <= 0.0:
            raise ValueError(f"clip_param must be positive, got {self.clip_param}.")
        if self.ppo_entropy_coef < 0.0:
            raise ValueError(f"ppo_entropy_coef must be non-negative, got {self.ppo_entropy_coef}.")
        if self.ppo_value_loss_coef < 0.0:
            raise ValueError(f"ppo_value_loss_coef must be non-negative, got {self.ppo_value_loss_coef}.")
        if self.ppo_schedule not in {"adaptive", "fixed"}:
            raise ValueError(f"Unsupported ppo_schedule={self.ppo_schedule!r}. Expected 'adaptive' or 'fixed'.")
        if not 0.0 <= self.cosine_weight_min <= self.cosine_weight_max <= 1.0:
            raise ValueError(
                "cosine_weight_min and cosine_weight_max must satisfy "
                f"0 <= min <= max <= 1, got {self.cosine_weight_min}, {self.cosine_weight_max}."
            )
        if self.cosine_weight_power <= 0.0:
            raise ValueError(f"cosine_weight_power must be positive, got {self.cosine_weight_power}.")

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        value_actions = self.policy.action_mean.detach() if self.actor_route in {"hybrid", "ppo_only"} else self.transition.actions
        self.transition.values = self.policy.evaluate(obs, value_actions).detach()
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
        if self.actor_route in {"hybrid", "ppo_only"}:
            last_action = self.policy.action_mean.detach()
        last_values = self.policy.evaluate(obs, last_action).detach().view(-1, 1)
        if self.actor_route in {"hybrid", "ppo_only"}:
            self._compute_ppo_returns(last_values)
            return

        recurr_value = last_values
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta_1 = next_is_not_terminal * self.gamma * next_values
            delta_n = next_is_not_terminal * self.gamma * recurr_value
            recurr_value = st.soft_rewards[step] + (1 - self.lam) * delta_1 + self.lam * delta_n
            st.returns[step] = recurr_value
        st.advantages = st.returns - st.values
        if not self.ppo_advantage_normalization:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def _compute_ppo_returns(self, last_values: torch.Tensor) -> None:
        st = self.storage
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.ppo_advantage_normalization:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        if self.actor_route in {"hybrid", "ppo_only"}:
            return self._update_actor_route_with_ppo_mechanics()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_actor_reppo_loss = 0.0
        mean_actor_ppo_loss = 0.0
        mean_reppo_route_weight = 0.0
        mean_cosine_gq_gppo = 0.0
        mean_norm_gq = 0.0
        mean_norm_gppo = 0.0
        mean_ppo_advantage_abs = 0.0
        mean_ratio_clip_fraction = 0.0
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
            values_batch,
            advantages_batch,
            returns_batch,
            truncations_batch,
            old_actions_log_prob_batch,
            old_mean_batch,
            old_std_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator(self.num_mini_batches, self.num_learning_epochs):
            actor_metrics = self.update_actor(
                {
                    "obs_batch": obs_batch,
                    "actions_batch": actions_batch,
                    "values_batch": values_batch,
                    "advantages_batch": advantages_batch,
                    "truncations_batch": truncations_batch,
                    "old_actions_log_prob_batch": old_actions_log_prob_batch,
                    "hidden_states_batch": hidden_states_batch,
                    "masks_batch": masks_batch,
                    "old_mean": old_mean_batch,
                    "old_std": old_std_batch,
                }
            )
            mean_entropy += actor_metrics["entropy"]
            mean_surrogate_loss += actor_metrics["actor_loss"]
            mean_actor_reppo_loss += actor_metrics["actor_loss_reppo"]
            mean_actor_ppo_loss += actor_metrics["actor_loss_ppo"]
            mean_reppo_route_weight += actor_metrics["reppo_route_weight"]
            mean_cosine_gq_gppo += actor_metrics["cosine_gQ_gPPO"]
            mean_norm_gq += actor_metrics["norm_gQ"]
            mean_norm_gppo += actor_metrics["norm_gPPO"]
            mean_ppo_advantage_abs += actor_metrics["ppo_advantage_abs_mean"]
            mean_ratio_clip_fraction += actor_metrics["ratio_clip_fraction"]

        print("value prediction error: ", critic_metrics["value_prediction_error"])
        print("enc dec error: ", critic_metrics["enc_dec_error"])
        print("on policy values mean: ", actor_metrics["on_policy_values_mean"])
        print("entropy: ", actor_metrics["entropy"])
        print("kl divergence: ", actor_metrics["kl_divergence"])
        print("actor route: ", self.actor_route)
        print("reppo route weight: ", actor_metrics["reppo_route_weight"])
        print("cosine gQ gPPO: ", actor_metrics["cosine_gQ_gPPO"])
        print("entropy target: ", self.target_entropy)
        print("alpha temp: ", self.policy.alpha_temp.item())
        print("alpha kl: ", self.policy.alpha_kl.item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_actor_reppo_loss /= num_updates
        mean_actor_ppo_loss /= num_updates
        mean_reppo_route_weight /= num_updates
        mean_cosine_gq_gppo /= num_updates
        mean_norm_gq /= num_updates
        mean_norm_gppo /= num_updates
        mean_ppo_advantage_abs /= num_updates
        mean_ratio_clip_fraction /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "actor_reppo": mean_actor_reppo_loss,
            "actor_ppo": mean_actor_ppo_loss,
            "reppo_route_weight": mean_reppo_route_weight,
            "cosine_gQ_gPPO": mean_cosine_gq_gppo,
            "norm_gQ": mean_norm_gq,
            "norm_gPPO": mean_norm_gppo,
            "ppo_advantage_abs_mean": mean_ppo_advantage_abs,
            "ratio_clip_fraction": mean_ratio_clip_fraction,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def _update_actor_route_with_ppo_mechanics(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_actor_reppo_loss = 0.0
        mean_actor_ppo_loss = 0.0
        mean_reppo_route_weight = 0.0
        mean_cosine_gq_gppo = 0.0
        mean_norm_gq = 0.0
        mean_norm_gppo = 0.0
        mean_ppo_advantage_abs = 0.0
        mean_ratio_clip_fraction = 0.0
        mean_kl_divergence = 0.0
        mean_value_prediction_error = 0.0
        mean_enc_dec_error = 0.0
        mean_on_policy_values = 0.0

        with torch.no_grad():
            self.old_policy.load_state_dict(self.policy.state_dict())

        generator = self.storage.recurrent_mini_batch_generator if self.policy.is_recurrent else self.storage.mini_batch_generator
        for (
            obs_batch,
            actions_batch,
            values_batch,
            advantages_batch,
            returns_batch,
            truncations_batch,
            old_actions_log_prob_batch,
            old_mean_batch,
            old_std_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator(self.num_mini_batches, self.num_learning_epochs):
            del values_batch

            self.policy.act(obs_batch, hidden_states_batch, masks_batch)
            predicted_policy = self.policy.distribution
            predicted_actions = predicted_policy.rsample()
            entropy = self._policy_entropy(predicted_policy, predicted_actions)

            self._set_critic_grad(False)
            on_policy_values = self.policy.evaluate(obs_batch, predicted_actions, hidden_states_batch, masks_batch)
            pathwise_q_loss = -on_policy_values.mean()
            self._set_critic_grad(True)

            value_loss, value_metrics = self._compute_value_loss(
                obs_batch,
                old_mean_batch,
                returns_batch,
                truncations_batch,
                hidden_states_batch,
                masks_batch,
            )

            ppo_kl_divergence = self._compute_ppo_kl_divergence(
                predicted_policy,
                obs_batch,
                hidden_states_batch,
                masks_batch,
            )
            self._adapt_ppo_learning_rate(ppo_kl_divergence.mean())

            reppo_policy_loss = pathwise_q_loss if self.actor_route == "hybrid" else torch.tensor(0.0, device=self.device)

            ppo_policy_loss, ppo_surrogate_loss, ppo_metrics = self._compute_ppo_actor_loss(
                predicted_policy,
                entropy,
                actions_batch,
                advantages_batch,
                old_actions_log_prob_batch,
            )

            cosine_gq_gppo = torch.tensor(0.0, device=self.device)
            norm_gq = torch.tensor(0.0, device=self.device)
            norm_gppo = torch.tensor(0.0, device=self.device)
            reppo_route_weight = torch.tensor(0.0, device=self.device)
            policy_loss = ppo_policy_loss
            if self.actor_route == "hybrid":
                cosine_gq_gppo, norm_gq, norm_gppo = self._actor_loss_cosine(pathwise_q_loss, ppo_surrogate_loss)
                reppo_route_weight = self._cosine_to_reppo_weight(cosine_gq_gppo)
                policy_loss = reppo_route_weight * reppo_policy_loss + (1.0 - reppo_route_weight) * ppo_policy_loss

            self.optimizer.zero_grad()
            (self.ppo_value_loss_coef * value_loss).backward()
            self._set_critic_grad(False)
            actor_loss = policy_loss
            actor_loss.backward()
            self._set_critic_grad(True)
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += ppo_surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_actor_reppo_loss += reppo_policy_loss.item()
            mean_actor_ppo_loss += ppo_policy_loss.item()
            mean_reppo_route_weight += reppo_route_weight.item()
            mean_cosine_gq_gppo += cosine_gq_gppo.item()
            mean_norm_gq += norm_gq.item()
            mean_norm_gppo += norm_gppo.item()
            mean_ppo_advantage_abs += ppo_metrics["advantage_abs_mean"]
            mean_ratio_clip_fraction += ppo_metrics["ratio_clip_fraction"]
            mean_kl_divergence += ppo_kl_divergence.mean().item()
            mean_value_prediction_error += value_metrics["value_prediction_error"]
            mean_enc_dec_error += value_metrics["enc_dec_error"]
            mean_on_policy_values += on_policy_values.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_actor_reppo_loss /= num_updates
        mean_actor_ppo_loss /= num_updates
        mean_reppo_route_weight /= num_updates
        mean_cosine_gq_gppo /= num_updates
        mean_norm_gq /= num_updates
        mean_norm_gppo /= num_updates
        mean_ppo_advantage_abs /= num_updates
        mean_ratio_clip_fraction /= num_updates
        mean_kl_divergence /= num_updates
        mean_value_prediction_error /= num_updates
        mean_enc_dec_error /= num_updates
        mean_on_policy_values /= num_updates

        print("value prediction error: ", mean_value_prediction_error)
        print("enc dec error: ", mean_enc_dec_error)
        print("on policy values mean: ", mean_on_policy_values)
        print("entropy: ", mean_entropy)
        print("kl divergence: ", mean_kl_divergence)
        print("actor route: ", self.actor_route)
        print("reppo route weight: ", mean_reppo_route_weight)
        print("cosine gQ gPPO: ", mean_cosine_gq_gppo)
        print("entropy target: ", self.target_entropy)
        print("alpha temp: ", self.policy.alpha_temp.item())
        print("alpha kl: ", self.policy.alpha_kl.item())

        self.storage.clear()
        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "actor_reppo": mean_actor_reppo_loss,
            "actor_ppo": mean_actor_ppo_loss,
            "reppo_route_weight": mean_reppo_route_weight,
            "cosine_gQ_gPPO": mean_cosine_gq_gppo,
            "norm_gQ": mean_norm_gq,
            "norm_gPPO": mean_norm_gppo,
            "ppo_advantage_abs_mean": mean_ppo_advantage_abs,
            "ratio_clip_fraction": mean_ratio_clip_fraction,
        }

    def update_actor(self, minibatch: dict) -> dict:
        obs_batch = minibatch["obs_batch"]
        actions_batch = minibatch["actions_batch"]
        advantages_batch = minibatch["advantages_batch"]
        old_actions_log_prob_batch = minibatch["old_actions_log_prob_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        self.policy.act(obs_batch, hidden_states_batch, masks_batch)
        predicted_policy = self.policy.distribution
        predicted_actions = predicted_policy.rsample()
        on_policy_values = self.policy.evaluate(obs_batch, predicted_actions, hidden_states_batch, masks_batch)

        entropy = self._policy_entropy(predicted_policy, predicted_actions)
        entropy_loss = self.policy.alpha_temp.detach() * entropy
        primary_policy_loss = -(on_policy_values + entropy_loss)
        pathwise_q_loss = -on_policy_values.mean()

        kl_divergence = self._compute_reppo_kl_divergence(
            predicted_policy,
            obs_batch,
            hidden_states_batch,
            masks_batch,
        )
        if self.actor_route == "ppo_only":
            self._adapt_ppo_learning_rate(kl_divergence.mean())

        policy_loss = torch.where(
            (kl_divergence < self.desired_kl).detach(),
            primary_policy_loss,
            self.policy.alpha_kl.detach() * kl_divergence,
        ).mean()
        reppo_policy_loss = policy_loss

        ppo_policy_loss, ppo_surrogate_loss, ppo_metrics = self._compute_ppo_actor_loss(
            predicted_policy,
            entropy,
            actions_batch,
            advantages_batch,
            old_actions_log_prob_batch,
        )
        cosine_gq_gppo = torch.tensor(0.0, device=self.device)
        norm_gq = torch.tensor(0.0, device=self.device)
        norm_gppo = torch.tensor(0.0, device=self.device)
        reppo_route_weight = torch.tensor(1.0, device=self.device)
        if self.actor_route in {"hybrid", "ppo_only"}:
            cosine_gq_gppo, norm_gq, norm_gppo = self._actor_loss_cosine(pathwise_q_loss, ppo_surrogate_loss)
        if self.actor_route == "hybrid":
            reppo_route_weight = self._cosine_to_reppo_weight(cosine_gq_gppo)
            policy_loss = reppo_route_weight * policy_loss + (1.0 - reppo_route_weight) * ppo_policy_loss
        elif self.actor_route == "ppo_only":
            reppo_route_weight = torch.tensor(0.0, device=self.device)
            policy_loss = ppo_policy_loss

        self._set_critic_grad(False)
        self.optimizer.zero_grad()
        actor_loss = policy_loss
        if self.actor_route != "ppo_only":
            temp_target_loss = self.policy.alpha_temp * (entropy.mean() - self.target_entropy).detach()
            kl_target_loss = self.policy.alpha_kl * (self.desired_kl - kl_divergence.mean()).detach()
            actor_loss = actor_loss + temp_target_loss
            actor_loss = actor_loss + kl_target_loss
        actor_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._set_critic_grad(True)

        return {
            "actor_loss": actor_loss.item(),
            "actor_loss_reppo": reppo_policy_loss.item(),
            "actor_loss_ppo": ppo_policy_loss.item(),
            "reppo_route_weight": reppo_route_weight.item(),
            "cosine_gQ_gPPO": cosine_gq_gppo.item(),
            "norm_gQ": norm_gq.item(),
            "norm_gPPO": norm_gppo.item(),
            "ppo_advantage_abs_mean": ppo_metrics["advantage_abs_mean"],
            "ratio_clip_fraction": ppo_metrics["ratio_clip_fraction"],
            "entropy": entropy.mean().item(),
            "kl_divergence": kl_divergence.mean().item(),
            "on_policy_values_mean": on_policy_values.mean().item(),
        }

    def _policy_entropy(
        self, predicted_policy: torch.distributions.Distribution, predicted_actions: torch.Tensor
    ) -> torch.Tensor:
        if getattr(self.policy, "distribution_type", None) == "normal":
            entropy_attr = getattr(predicted_policy, "entropy", None)
            if callable(entropy_attr):
                try:
                    entropy = entropy_attr()
                except NotImplementedError:
                    entropy = None
                if entropy is not None:
                    return entropy.sum(-1) if entropy.ndim > 1 else entropy
        return -predicted_policy.log_prob(predicted_actions).sum(-1)

    def _compute_reppo_kl_divergence(
        self,
        predicted_policy: torch.distributions.Distribution,
        obs_batch: TensorDict,
        hidden_states_batch: torch.Tensor,
        masks_batch: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            self.old_policy.act(obs_batch, hidden_states_batch, masks_batch)
            old_policy_distribution = self.old_policy.distribution
            old_policy_actions = old_policy_distribution.sample((4,))
            log_prob_old = old_policy_distribution.log_prob(old_policy_actions).detach()
        log_prob_new = predicted_policy.log_prob(old_policy_actions)
        return (log_prob_old - log_prob_new).sum(-1).mean(0)

    def _compute_ppo_kl_divergence(
        self,
        predicted_policy: torch.distributions.Distribution,
        obs_batch: TensorDict,
        hidden_states_batch: torch.Tensor,
        masks_batch: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            self.old_policy.act(obs_batch, hidden_states_batch, masks_batch)
            old_policy_distribution = self.old_policy.distribution
            old_loc, old_scale = self._base_normal_params(old_policy_distribution)

        new_loc, new_scale = self._base_normal_params(predicted_policy)
        if old_loc is None or old_scale is None or new_loc is None or new_scale is None:
            return self._compute_reppo_kl_divergence(
                predicted_policy,
                obs_batch,
                hidden_states_batch,
                masks_batch,
            ).detach()

        old_scale = old_scale.detach().clamp_min(1.0e-8)
        old_loc = old_loc.detach()
        new_scale = new_scale.clamp_min(1.0e-8)
        kl = torch.log(new_scale / old_scale)
        kl = kl + (old_scale.square() + (old_loc - new_loc).square()) / (2.0 * new_scale.square())
        kl = kl - 0.5
        return kl.sum(-1)

    @staticmethod
    def _base_normal_params(distribution: torch.distributions.Distribution) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        base_distribution = getattr(distribution, "base_dist", distribution)
        loc = getattr(base_distribution, "loc", None)
        scale = getattr(base_distribution, "scale", None)
        return loc, scale

    def _compute_ppo_actor_loss(
        self,
        predicted_policy: torch.distributions.Distribution,
        entropy: torch.Tensor,
        actions_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            advantages = advantages_batch.view(-1)
            advantage_mean = advantages.mean()
            advantage_std = advantages.std(unbiased=False)
            advantage_abs_mean = advantages.abs().mean()
            if self.ppo_advantage_normalization:
                advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

        actions_log_prob = predicted_policy.log_prob(actions_batch).sum(-1).view(-1)
        old_actions_log_prob = old_actions_log_prob_batch.view(-1)
        ratio = torch.exp(actions_log_prob - old_actions_log_prob)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        policy_loss = surrogate_loss - self.ppo_entropy_coef * entropy.mean()

        with torch.no_grad():
            ratio_clip_fraction = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).float().mean()
            ratio_std = ratio.std(unbiased=False)
        return (
            policy_loss,
            surrogate_loss,
            {
                "advantage_mean": advantage_mean.item(),
                "advantage_std": advantage_std.item(),
                "advantage_abs_mean": advantage_abs_mean.item(),
                "ratio_mean": ratio.mean().item(),
                "ratio_std": ratio_std.item(),
                "ratio_clip_fraction": ratio_clip_fraction.item(),
            },
        )

    def _adapt_ppo_learning_rate(self, kl_mean: torch.Tensor) -> None:
        if self.desired_kl is None or self.ppo_schedule != "adaptive":
            return
        with torch.inference_mode():
            kl_mean = kl_mean.detach()
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def update_critic(self, minibatch: dict) -> dict:
        obs_batch = minibatch["obs_batch"]
        actions_batch = minibatch["actions_batch"]
        returns_batch = minibatch["returns_batch"]
        truncations_batch = minibatch["truncations_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        value_loss, value_metrics = self._compute_value_loss(
            obs_batch,
            actions_batch,
            returns_batch,
            truncations_batch,
            hidden_states_batch,
            masks_batch,
        )

        self.optimizer.zero_grad()
        value_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {"value_loss": value_loss.item(), **value_metrics}

    def _compute_value_loss(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        truncations_batch: torch.Tensor,
        hidden_states_batch: torch.Tensor,
        masks_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        value_prediction, value_logits = self.policy.evaluate(
            obs_batch, actions_batch, hidden_states_batch, masks_batch, return_logits=True
        )
        embedded_returns = self.policy.hlgauss_embed(returns_batch.view(-1)).view(value_logits.shape).detach()
        value_loss_per_sample = -(embedded_returns * torch.log_softmax(value_logits, dim=-1)).sum(-1)
        value_loss_weights = 1.0 - truncations_batch.view(-1).float()
        value_loss = (value_loss_weights * value_loss_per_sample).sum() / value_loss_weights.sum().clamp_min(1.0)

        value_prediction_error = (value_prediction.view(-1) - returns_batch.view(-1)).abs().mean().item()
        decoded_returns = self.policy.hlgauss_decode(torch.log(embedded_returns)).detach()
        enc_dec_error = (decoded_returns.view(-1) - returns_batch.view(-1)).abs().mean().item()
        return value_loss, {
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

    def _actor_loss_cosine(self, first_loss: torch.Tensor, second_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_params = [param for param in self.policy.actor.parameters() if param.requires_grad]
        if not actor_params:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero, zero

        first_grads = torch.autograd.grad(first_loss, actor_params, retain_graph=True, allow_unused=True)
        second_grads = torch.autograd.grad(second_loss, actor_params, retain_graph=True, allow_unused=True)
        first_flat = self._flatten_grads(first_grads, actor_params)
        second_flat = self._flatten_grads(second_grads, actor_params)
        first_norm = first_flat.norm()
        second_norm = second_flat.norm()
        cosine = torch.dot(first_flat, second_flat) / (first_norm * second_norm + 1e-8)
        return cosine.detach(), first_norm.detach(), second_norm.detach()

    def _cosine_to_reppo_weight(self, cosine: torch.Tensor) -> torch.Tensor:
        normalized = cosine.clamp(0.0, 1.0).pow(self.cosine_weight_power)
        return self.cosine_weight_min + (self.cosine_weight_max - self.cosine_weight_min) * normalized

    @staticmethod
    def _flatten_grads(
        grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]
    ) -> torch.Tensor:
        flat_grads = [
            torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1)
            for grad, param in zip(grads, params, strict=True)
        ]
        if not flat_grads:
            return torch.empty(0)
        return torch.cat(flat_grads)

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
