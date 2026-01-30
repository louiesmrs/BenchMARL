from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch
from torchrl.data.tensor_specs import Composite, Categorical, UnboundedContinuous, UnboundedDiscrete
from torchrl.envs import EnvBase

from benchmarl.environments.common import TaskClass


@dataclass
class TaskConfig:
    curriculum_path: str
    map_width: int = 30
    map_height: int = 30
    num_agents: int = 10
    num_steps: int = 200


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
            from flatland_torchrl.torchrl_rail_env import TDRailEnv, TorchRLRailEnv

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
            )
            td_env.reset()
            env = TorchRLRailEnv(td_env)
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
        num_agents = self.config.get("num_agents")
        obs_spec = Composite(
            agents=Composite(
                observation=Composite(
                    agents_attr=UnboundedContinuous(
                        shape=[num_agents, 83], dtype=torch.float32
                    ),
                    adjacency=UnboundedDiscrete(
                        shape=[num_agents, 30, 3], dtype=torch.int64
                    ),
                    node_attr=UnboundedDiscrete(
                        shape=[num_agents, 31, 12], dtype=torch.float32
                    ),
                    node_order=UnboundedDiscrete(
                        shape=[num_agents, 31], dtype=torch.int64
                    ),
                    edge_order=UnboundedDiscrete(
                        shape=[num_agents, 30], dtype=torch.int64
                    ),
                    valid_actions=Categorical(
                        n=2, shape=[num_agents, 5], dtype=torch.bool
                    ),
                    shape=[num_agents],
                ),
                shape=[],
            ),
            shape=[],
        )
        return obs_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        num_agents = self.config.get("num_agents")
        return Composite(
            agents=Composite(
                action=Categorical(n=5, shape=[num_agents], dtype=torch.int64),
                shape=[],
            ),
            shape=[],
        )

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        num_agents = self.config.get("num_agents")
        return Composite(
            agents=Composite(
                action_mask=Categorical(n=2, shape=[num_agents, 5], dtype=torch.bool),
                shape=[],
            ),
            shape=[],
        )


class FlatlandTask(Enum):
    PHASE_1_3_7_TO_10_AGENTS = ("phase_1_3_7_to_10_agents", TaskConfig)

    @staticmethod
    def env_name() -> str:
        return "flatland"

    @classmethod
    def _task_class(cls):
        return FlatlandClass

    @classmethod
    def _task_config_class(cls, task_name: str):
        return TaskConfig
