# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import FastTD3
from rsl_rl.env import VecEnv
from rsl_rl.models import FastTD3Actor
from rsl_rl.utils import check_nan
from rsl_rl.utils.logger import Logger


class OffPolicyRunner:
    """Off-policy runner for replay-buffer-based algorithms."""

    alg: FastTD3

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self._configure_multi_gpu()
        self.cfg.setdefault("algorithm", {})
        self.cfg["algorithm"].setdefault("rnd_cfg", None)
        self.cfg["algorithm"].setdefault("symmetry_cfg", None)

        obs = self.env.get_observations()
        self.alg = FastTD3.construct_algorithm(obs, self.env, self.cfg, self.device)
        self.alg.policy = self.alg.actor  # type: ignore[attr-defined]

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
        dones = torch.zeros(self.env.num_envs, device=self.device)
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # FastTD3 updates normalization buffers during rollout collection, so use no_grad
            # instead of inference_mode to keep checkpoint state reloadable in-process.
            with torch.no_grad():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, dones=dones)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras)

                stop = time.time()
                collect_time = stop - start
                start = stop

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            action_std = getattr(self.alg.get_policy(), "output_std", torch.zeros(1, device=self.device))
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=action_std,
                rnd_weight=None,
            )

            if self.gpu_global_rank == 0 and self.logger.log_dir is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.gpu_global_rank == 0 and self.logger.log_dir is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
        self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos") or {}

    def get_inference_policy(self, device: str | None = None) -> FastTD3Actor:
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)  # type: ignore[return-value]

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        policy = self.alg.get_policy().as_jit().to("cpu")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        traced_model = torch.jit.script(policy)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        policy = self.alg.get_policy().as_onnx(verbose=verbose).to("cpu")
        policy.eval()
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        torch.onnx.export(
            policy,
            policy.get_dummy_inputs(),  # type: ignore[attr-defined]
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=policy.input_names,  # type: ignore[attr-defined]
            output_names=policy.output_names,  # type: ignore[attr-defined]
        )

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
