from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from tensordict import TensorDictBase

import torch
from torchrl.data.tensor_specs import Composite
from torchrl.envs import EnvBase, RewardSum

from benchmarl.environments.common import Task, TaskClass


@dataclass
class TaskConfig:
    map_width: int = 30
    map_height: int = 30
    num_agents: int = 10
    num_steps: int = 200
    reward_coefs: Optional[dict] = None
    smoke_obs_only: bool = False


class FlatlandClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: str,
    ):
        if continuous_actions:
            raise ValueError("Flatland task only supports discrete actions")

        def make_env():
            from flatland_cutils import TreeObsForRailEnv as TreeCutils
            from flatland.envs.line_generators import SparseLineGen
            from flatland.envs.malfunction_generators import (
                MalfunctionParameters,
                ParamMalfunctionGen,
            )
            from flatland.envs.rail_generators import SparseRailGen
            from benchmarl.environments.flatland.torchrl_rail_env import (
                TDRailEnv,
                TorchRLRailEnv,
            )

            td_env = TDRailEnv(
                number_of_agents=self.config.get("num_agents"),
                width=self.config.get("map_width"),
                height=self.config.get("map_height"),
                rail_generator=SparseRailGen(
                    max_num_cities=3,
                    grid_mode=False,
                    max_rails_between_cities=2,
                    max_rail_pairs_in_city=2,
                ),
                line_generator=SparseLineGen(speed_ratio_map={1.0: 1}),
                malfunction_generator=ParamMalfunctionGen(
                    MalfunctionParameters(
                        malfunction_rate=1 / 4500, min_duration=20, max_duration=50
                    )
                ),
                obs_builder_object=TreeCutils(31, 500),
                obs_only=self.config.get("smoke_obs_only", False),
            )
            td_env.set_reward_coef(self.config.get("reward_coefs"))
            td_env.reset()
            env = TorchRLRailEnv(td_env, obs_only=self.config.get("smoke_obs_only", False))
            return env

        return make_env

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("num_steps", 200)

    def has_render(self, env: EnvBase) -> bool:
        return False

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        num_agents = self.config.get("num_agents")
        return {"agents": [str(i) for i in range(num_agents)]}

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.observation_spec.clone()

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.action_spec.clone()

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def get_reward_sum_transform(self, env: EnvBase):
        return RewardSum(in_keys=[("agents", "reward")], out_keys=["episode_reward"])

    @staticmethod
    def env_name() -> str:
        return "flatland"

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        if ("next", "agents", "observation", "agents_attr") not in batch.keys(True, True):
            return {}
        done = batch.get(("next", "done")).squeeze(-1)
        if done.numel() == 0 or not done.any():
            return {}
        final_stats = batch.get(("next", "agents", "observation", "agents_attr"))[done]
        arrival_ratio = final_stats[..., 6].mean().item()
        deadlock_ratio = final_stats[..., 41].mean().item()
        return {
            "collection/info/arrival_ratio": arrival_ratio,
            "collection/info/deadlock_ratio": deadlock_ratio,
        }


class FlatlandTask(Task):
    """Enum for Flatland tasks."""

    PHASE_1_3_7_TO_10_AGENTS = None
    PHASE_1 = None
    PHASE_3 = None
    PHASE_7 = None
    PHASE_10 = None
    SMOKE_2_AGENTS = None

    @staticmethod
    def associated_class():
        return FlatlandClass
