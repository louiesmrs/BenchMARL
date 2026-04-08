from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState
from tensordict.tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete, Categorical
from torchrl.envs.common import EnvBase

from benchmarl.environments.flatland.python_tree_adapter import FlatlandPythonTreeAdapter

RewardCoefs = NamedTuple(
    "RewardCoefs",
    [
        ("delay_reward", int),
        ("shortest_path_reward", int),
        ("arrival_reward", int),
        ("deadlock_penalty", int),
        ("departure_reward", int),
        ("arrival_delay_penalty", int),
    ],
)


class FlatlandRailEnv(RailEnv):
    """RailEnv wrapper that accepts/returns TensorDicts and supports reward coefs."""

    def __init__(
        self,
        *args,
        observation_backend: str = "flatland_cutils",
        tree_num_nodes: int = 31,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reward_coefs: Optional[RewardCoefs] = None
        self.previous_deadlocked: set
        self.fixed_env: bool = False
        self.observation_backend = observation_backend
        self.tree_adapter = FlatlandPythonTreeAdapter(max_nodes=tree_num_nodes)

    @classmethod
    def from_env(
        cls,
        env: RailEnv,
        fixed_env: bool = True,
        observation_backend: str = "flatland_cutils",
        tree_num_nodes: int = 31,
    ):
        obj = cls.__new__(cls)
        obj.__dict__ = env.__dict__
        obj.reward_coefs = None
        obj.previous_deadlocked = set()
        obj.fixed_env = fixed_env
        obj.observation_backend = observation_backend
        obj.tree_adapter = FlatlandPythonTreeAdapter(max_nodes=tree_num_nodes)
        return obj

    def set_reward_coef(self, reward_coefs: Optional[dict]) -> None:
        if reward_coefs is None:
            self.reward_coefs = None
            return
        self.reward_coefs = RewardCoefs(**reward_coefs)

    def obs_to_td(self, observations) -> tuple[TensorDict, torch.Tensor]:
        if self.observation_backend == "flatland_cutils":
            obs_td: TensorDict = TensorDict(
                {
                    "agents_attr": torch.tensor(observations[0], dtype=torch.float32),
                    "node_attr": torch.tensor(observations[1][0], dtype=torch.float32),
                    "adjacency": torch.tensor(observations[1][1], dtype=torch.int64),
                    "node_order": torch.tensor(observations[1][2], dtype=torch.int64),
                    "edge_order": torch.tensor(observations[1][3], dtype=torch.int64),
                },
                [self.get_num_agents()],
            )
            _, _, valid_actions = self.obs_builder.get_properties()
            valid_actions = torch.tensor(valid_actions, dtype=torch.bool)
            return obs_td, valid_actions

        (
            agents_attr,
            node_attr,
            adjacency,
            node_order,
            edge_order,
            valid_actions,
        ) = self.tree_adapter.encode(self, observations)
        obs_td = TensorDict(
            {
                "agents_attr": torch.tensor(agents_attr, dtype=torch.float32),
                "node_attr": torch.tensor(node_attr, dtype=torch.float32),
                "adjacency": torch.tensor(adjacency, dtype=torch.int64),
                "node_order": torch.tensor(node_order, dtype=torch.int64),
                "edge_order": torch.tensor(edge_order, dtype=torch.int64),
            },
            [self.get_num_agents()],
        )
        return obs_td, torch.tensor(valid_actions, dtype=torch.bool)

    def reset(
        self,
        regenerate_rail: bool = True,
        regenerate_schedule: bool = True,
        *,
        random_seed: int = None,
    ) -> TensorDict:
        if getattr(self, "fixed_env", False):
            regenerate_rail = False
            regenerate_schedule = False
        observations, _ = super().reset(
            regenerate_rail=regenerate_rail,
            regenerate_schedule=regenerate_schedule,
            random_seed=random_seed,
        )
        tensordict_out: TensorDict = TensorDict({}, batch_size=[])
        tensordict_out["agents"] = TensorDict({}, batch_size=[])
        obs_td, valid_actions = self.obs_to_td(observations)
        tensordict_out["agents"]["observation"] = obs_td
        tensordict_out["agents"]["observation"]["valid_actions"] = valid_actions
        self.previous_deadlocked = self.motionCheck.svDeadlocked
        return tensordict_out

    def step(self, tensordict: TensorDict) -> TensorDict:
        actions = {
            handle: action.item()
            for handle, action in enumerate(tensordict["agents"]["action"].flatten())
        }
        observations, rewards, done, _ = super().step(actions)
        return_td: TensorDict = TensorDict({"agents": TensorDict({}, [])}, batch_size=[])
        obs_td, valid_actions = self.obs_to_td(observations)
        return_td["agents"]["observation"] = obs_td
        return_td["agents"]["reward"] = torch.tensor(
            [value for _, value in rewards.items()], dtype=torch.float32
        ).unsqueeze(-1)
        global_done = torch.tensor(done["__all__"]).type(torch.bool)
        return_td["done"] = global_done.view(1)
        return_td["terminated"] = global_done.view(1)
        return_td["truncated"] = torch.tensor(False)
        return_td["agents"]["observation"]["valid_actions"] = valid_actions
        return return_td

    def update_step_rewards(self, i_agent: int) -> None:
        if self.reward_coefs is None:
            return

        agent: EnvAgent = self.agents[i_agent]
        delay_reward = 0
        shortest_path_reward = 0
        arrival_reward = 0
        deadlock_penalty = 0
        departure_reward = 0
        arrival_delay_penalty = 0

        if self.reward_coefs.delay_reward != 0:
            if (
                agent.earliest_departure <= self._elapsed_steps
                and agent.state != TrainState.DONE
            ):
                delay_reward = min(
                    agent.get_current_delay(self._elapsed_steps, self.distance_map), 0
                )

        if self.reward_coefs.shortest_path_reward != 0:
            if (
                agent.earliest_departure <= self._elapsed_steps
                and agent.state != TrainState.DONE
            ):
                shortest_path_reward = agent.get_current_delay(
                    self._elapsed_steps, self.distance_map
                )

        if self.reward_coefs.arrival_reward != 0:
            if (
                agent.state == TrainState.DONE
                and agent.state_machine.previous_state != TrainState.DONE
                and self._elapsed_steps <= agent.latest_arrival
            ):
                arrival_reward = 1

        if self.reward_coefs.deadlock_penalty != 0:
            if (agent.position in self.motionCheck.svDeadlocked) and (
                agent.position not in self.previous_deadlocked
            ):
                deadlock_penalty = -1

        if self.reward_coefs.departure_reward != 0:
            if (
                agent.state == TrainState.MOVING
                and agent.state_machine.previous_state == TrainState.READY_TO_DEPART
            ):
                departure_reward = 1

        if self.reward_coefs.arrival_delay_penalty != 0:
            if agent.state == TrainState.DONE:
                if agent.state_machine.previous_state != TrainState.DONE:
                    arrival_delay_penalty = min(
                        agent.get_current_delay(self._elapsed_steps, self.distance_map),
                        0,
                    )
            else:
                arrival_delay_penalty = 0

        self.rewards_dict[i_agent] = (
            self.reward_coefs.delay_reward * delay_reward
            + self.reward_coefs.shortest_path_reward * shortest_path_reward
            + self.reward_coefs.arrival_reward * arrival_reward
            + self.reward_coefs.deadlock_penalty * deadlock_penalty
            + self.reward_coefs.departure_reward * departure_reward
            + self.reward_coefs.arrival_delay_penalty * arrival_delay_penalty
        )

    def _handle_end_reward(self, agent: EnvAgent) -> int:
        if self.reward_coefs is None:
            return super()._handle_end_reward(agent)
        if agent.state == TrainState.DONE:
            return 0
        if self.reward_coefs.arrival_delay_penalty != 0:
            return min(
                agent.get_current_delay(self._elapsed_steps, self.distance_map), 0
            )
        return 0


class FlatlandTorchRLEnv(EnvBase):
    """TorchRL EnvBase wrapper for FlatlandRailEnv."""

    def __init__(self, env: FlatlandRailEnv):
        super().__init__()
        self.env = env
        self.num_agents = env.get_num_agents()
        self._make_spec()
        self.rng: Optional[int] = None

    def _set_seed(self, seed: Optional[int]) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _make_spec(self) -> None:
        self.observation_spec = Composite(
            agents=Composite(
                observation=Composite(
                    agents_attr=UnboundedContinuous(
                        shape=[self.num_agents, 83], dtype=torch.float32
                    ),
                    adjacency=UnboundedDiscrete(
                        shape=[self.num_agents, 30, 3], dtype=torch.int64
                    ),
                    node_attr=UnboundedDiscrete(
                        shape=[self.num_agents, 31, 12], dtype=torch.float32
                    ),
                    node_order=UnboundedDiscrete(
                        shape=[self.num_agents, 31], dtype=torch.int64
                    ),
                    edge_order=UnboundedDiscrete(
                        shape=[self.num_agents, 30], dtype=torch.int64
                    ),
                    valid_actions=Categorical(
                        n=2, shape=[self.num_agents, 5], dtype=torch.bool
                    ),
                    shape=[self.num_agents],
                ),
                shape=[self.num_agents],
            ),
            shape=[],
        )
        self.action_spec = Composite(
            agents=Composite(
                action=Categorical(n=5, shape=[self.num_agents], dtype=torch.int64),
                shape=[self.num_agents],
            ),
            shape=[],
        )
        self.reward_spec = Composite(
            agents=Composite(
                reward=UnboundedContinuous(
                    shape=[self.num_agents, 1], dtype=torch.float32
                ),
                shape=[self.num_agents],
            ),
            shape=[],
        )
        self.done_spec = Categorical(n=2, dtype=torch.bool, shape=[1])
        self.terminated_spec = Categorical(n=2, dtype=torch.bool, shape=[1])

    def _resolve_output_device(self, input_td: TensorDict | None = None):
        if input_td is not None:
            candidate_keys = (
                "done",
                "terminated",
                "truncated",
                "reset",
                ("agents", "done"),
                ("agents", "terminated"),
                ("agents", "reset"),
                ("next", "done"),
                ("next", "terminated"),
                ("next", "truncated"),
            )
            for key in candidate_keys:
                tensor = input_td.get(key, None)
                if isinstance(tensor, torch.Tensor):
                    return tensor.device

            if getattr(input_td, "device", None) is not None:
                return input_td.device

            for value in input_td.values(True, True):
                if isinstance(value, torch.Tensor):
                    return value.device

        return getattr(self, "device", None)

    @staticmethod
    def _to_device_if_needed(tensordict: TensorDict, device):
        if device is None:
            return tensordict
        return tensordict.to(device)

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:
        out = self.env.reset()
        return self._to_device_if_needed(
            out, self._resolve_output_device(input_td=tensordict)
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        out = self.env.step(tensordict)
        return self._to_device_if_needed(
            out, self._resolve_output_device(input_td=tensordict)
        )
