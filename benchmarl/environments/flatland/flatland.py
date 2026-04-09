from dataclasses import dataclass
from typing import Dict, List, Optional

from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase, RewardSum, Transform
from torchrl.envs.transforms import FlattenObservation

from benchmarl.environments.flatland.transforms import FlatlandTreePreprocessor

from benchmarl.environments.common import Task, TaskClass


@dataclass
class TaskConfig:
    map_width: int = 30
    map_height: int = 30
    num_agents: int = 2
    num_steps: int = 1000
    reward_coefs: Optional[dict] = None
    env_pickle: Optional[str] = None
    tree_observation_backend: str = "flatland_cutils"
    tree_num_nodes: int = 31
    tree_predictor_depth: int = 500
    tree_max_depth: int = 2


class FlatlandClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: str,
    ):
        _ = num_envs  # API contract: accepted for parity with other tasks.
        if continuous_actions:
            raise ValueError("Flatland task only supports discrete actions")

        def make_env():
            from flatland.envs.line_generators import SparseLineGen
            from flatland.envs.malfunction_generators import (
                MalfunctionParameters,
                ParamMalfunctionGen,
            )
            from flatland.envs.observations import TreeObsForRailEnv as TreePython
            from flatland.envs.persistence import RailEnvPersister
            from flatland.envs.rail_generators import SparseRailGen
            from benchmarl.environments.flatland.torchrl_flatland_env import (
                FlatlandRailEnv,
                FlatlandTorchRLEnv,
            )

            observation_backend = str(
                self.config.get("tree_observation_backend", "flatland_cutils")
            )
            tree_num_nodes = int(self.config.get("tree_num_nodes", 31))
            tree_predictor_depth = int(self.config.get("tree_predictor_depth", 500))
            tree_max_depth = int(self.config.get("tree_max_depth", 2))

            if observation_backend == "flatland_cutils":
                try:
                    from flatland_cutils import TreeObsForRailEnv as TreeCutils
                except ImportError as err:
                    raise ImportError(
                        "tree_observation_backend=flatland_cutils requires flatland_cutils to be installed. "
                        "Set tree_observation_backend=flatland to use the pure-python tree observation builder."
                    ) from err
                obs_builder = TreeCutils(tree_num_nodes, tree_predictor_depth)
            elif observation_backend == "flatland":
                obs_builder = TreePython(max_depth=tree_max_depth)
            else:
                raise ValueError(
                    "Unknown tree_observation_backend='"
                    f"{observation_backend}'. Expected one of: flatland_cutils, flatland"
                )

            env_pickle = self.config.get("env_pickle")
            if env_pickle:
                env, _ = RailEnvPersister.load_new(env_pickle)
                env.obs_builder = obs_builder
                env.obs_builder.set_env(env)
                td_env = FlatlandRailEnv.from_env(
                    env,
                    fixed_env=False,
                    observation_backend=observation_backend,
                    tree_num_nodes=tree_num_nodes,
                )
                td_env.set_reward_coef(self.config.get("reward_coefs"))
                td_env.reset(regenerate_rail=True, regenerate_schedule=True)
                torchrl_env = FlatlandTorchRLEnv(td_env)
                return torchrl_env.to(device) if device is not None else torchrl_env

            td_env = FlatlandRailEnv(
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
                obs_builder_object=obs_builder,
                observation_backend=observation_backend,
                tree_num_nodes=tree_num_nodes,
            )
            td_env.set_reward_coef(self.config.get("reward_coefs"))
            td_env.reset()
            torchrl_env = FlatlandTorchRLEnv(td_env)
            return torchrl_env.to(device) if device is not None else torchrl_env

        return make_env

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("num_steps", 1000)

    def has_render(self, env: EnvBase) -> bool:
        return False

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        num_agents = getattr(env, "num_agents", None) or self.config.get("num_agents")
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

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        return [
            FlatlandTreePreprocessor(agent_group="agents"),
            FlattenObservation(
                in_keys=[
                    ("agents", "observation", "adjacency"),
                    ("agents", "observation", "node_attr"),
                ],
                first_dim=-2,
                last_dim=-1,
            ),
        ]

    def get_reward_sum_transform(self, env: EnvBase):
        return RewardSum(
            in_keys=[("agents", "reward")], out_keys=[("agents", "episode_reward")]
        )

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

    FLATLAND = None
    FLATLAND_BASE = None

    @staticmethod
    def associated_class():
        return FlatlandClass
