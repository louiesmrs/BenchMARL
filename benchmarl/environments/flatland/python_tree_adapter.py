from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class FlatlandPythonTreeAdapter:
    """Converts Flatland python TreeObs nodes to BenchMARL tensor-friendly arrays.

    Output layout mirrors the cutils observation contract expected by the
    Flatland tree models:
      - agents_attr: [n_agents, agent_attr_size]
      - node_attr: [n_agents, max_nodes, node_attr_size]
      - adjacency: [n_agents, max_edges, 3] (parent, child, slot)
      - node_order: [n_agents, max_nodes]
      - edge_order: [n_agents, max_edges]
      - valid_actions: [n_agents, 5]
    """

    _CHILD_KEYS: tuple[str, ...] = ("L", "F", "R")
    _SLOT_MAP = {"L": -1, "F": 0, "R": 1}

    def __init__(
        self,
        max_nodes: int = 31,
        node_attr_size: int = 12,
        agent_attr_size: int = 83,
    ) -> None:
        self.max_nodes = max_nodes
        self.max_edges = max_nodes - 1
        self.node_attr_size = node_attr_size
        self.agent_attr_size = agent_attr_size

    def encode(self, env: Any, observations: dict[int, Any]):
        n_agents = env.get_num_agents()

        agents_attr = np.zeros((n_agents, self.agent_attr_size), dtype=np.float32)
        node_attr = np.zeros(
            (n_agents, self.max_nodes, self.node_attr_size), dtype=np.float32
        )
        adjacency = np.full((n_agents, self.max_edges, 3), -1, dtype=np.int64)
        node_order = np.zeros((n_agents, self.max_nodes), dtype=np.int64)
        edge_order = np.zeros((n_agents, self.max_edges), dtype=np.int64)
        valid_actions = np.zeros((n_agents, 5), dtype=np.bool_)

        arrival_ratio = self._arrival_ratio(env)
        deadlock_ratio = self._deadlock_ratio(env)

        for handle in range(n_agents):
            obs = observations.get(handle) if isinstance(observations, dict) else None
            (
                node_attr[handle],
                adjacency[handle],
                node_order[handle],
                edge_order[handle],
            ) = self._encode_tree(obs)
            agents_attr[handle] = self._encode_agent_attr(
                env,
                handle,
                arrival_ratio=arrival_ratio,
                deadlock_ratio=deadlock_ratio,
            )
            valid_actions[handle] = self._compute_valid_actions(env, handle)

        return agents_attr, node_attr, adjacency, node_order, edge_order, valid_actions

    def _encode_tree(self, root: Any):
        node_attr = np.zeros((self.max_nodes, self.node_attr_size), dtype=np.float32)
        adjacency = np.full((self.max_edges, 3), -1, dtype=np.int64)
        node_order = np.zeros((self.max_nodes,), dtype=np.int64)
        edge_order = np.zeros((self.max_edges,), dtype=np.int64)

        node_depth = np.full((self.max_nodes,), -1, dtype=np.int64)
        node_depth[0] = 0
        node_attr[0] = self._node_features(root)

        queue: deque[tuple[int, Any, int]] = deque([(0, root, 0)])
        next_node = 1
        next_edge = 0

        while queue and next_node < self.max_nodes and next_edge < self.max_edges:
            parent_idx, node, depth = queue.popleft()
            for label, child in self._iter_children(node):
                if next_node >= self.max_nodes or next_edge >= self.max_edges:
                    break
                child_idx = next_node
                next_node += 1

                adjacency[next_edge] = [parent_idx, child_idx, self._SLOT_MAP[label]]
                next_edge += 1

                node_depth[child_idx] = depth + 1
                node_attr[child_idx] = self._node_features(child)
                queue.append((child_idx, child, depth + 1))

        used_nodes = next_node
        if used_nodes > 0:
            max_depth = int(node_depth[:used_nodes].max())
            node_order[:used_nodes] = max_depth - node_depth[:used_nodes]

        for edge_idx in range(next_edge):
            parent = adjacency[edge_idx, 0]
            edge_order[edge_idx] = node_order[parent]

        return node_attr, adjacency, node_order, edge_order

    def _iter_children(self, node: Any):
        if node is None or not hasattr(node, "childs"):
            return
        childs = getattr(node, "childs", None)
        if not isinstance(childs, dict):
            return
        for key in self._CHILD_KEYS:
            child = childs.get(key, None)
            if self._is_missing_node(child):
                continue
            yield key, child

    @staticmethod
    def _is_missing_node(node: Any) -> bool:
        if node is None:
            return True
        if isinstance(node, (float, np.floating)) and np.isneginf(node):
            return True
        return False

    def _node_features(self, node: Any) -> np.ndarray:
        if self._is_missing_node(node) or not hasattr(node, "_fields"):
            return np.zeros((self.node_attr_size,), dtype=np.float32)

        values: list[float] = []
        for field in node._fields:
            if field == "childs":
                continue
            values.append(float(getattr(node, field)))

        arr = np.asarray(values, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.shape[0] < self.node_attr_size:
            pad = np.zeros((self.node_attr_size - arr.shape[0],), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > self.node_attr_size:
            arr = arr[: self.node_attr_size]

        return arr

    def _encode_agent_attr(
        self,
        env: Any,
        handle: int,
        *,
        arrival_ratio: float,
        deadlock_ratio: float,
    ) -> np.ndarray:
        vec = np.zeros((self.agent_attr_size,), dtype=np.float32)

        agent = env.agents[handle]
        width = max(1, int(getattr(env, "width", 1)))
        height = max(1, int(getattr(env, "height", 1)))

        if agent.state.is_off_map_state() and agent.initial_position is not None:
            pos = agent.initial_position
        elif agent.state.is_on_map_state() and agent.position is not None:
            pos = agent.position
        else:
            pos = agent.target

        if pos is None:
            pos = (0, 0)

        target = agent.target if agent.target is not None else (0, 0)

        vec[0] = float(pos[0]) / float(max(1, height - 1))
        vec[1] = float(pos[1]) / float(max(1, width - 1))
        vec[2] = float(target[0]) / float(max(1, height - 1))
        vec[3] = float(target[1]) / float(max(1, width - 1))
        vec[4] = float(getattr(agent, "direction", 0)) / 3.0
        vec[5] = float(getattr(agent.speed_counter, "speed", 1.0))
        vec[6] = arrival_ratio
        vec[7] = 1.0 if agent.state.is_on_map_state() else 0.0
        vec[8] = 1.0 if agent.state.is_off_map_state() else 0.0
        vec[9] = 1.0 if str(agent.state).endswith("DONE") else 0.0
        vec[10] = float(getattr(agent.malfunction_handler, "malfunction_down_counter", 0))
        vec[11] = float(getattr(agent, "earliest_departure", 0))
        vec[12] = float(getattr(agent, "latest_arrival", 0))
        vec[41] = deadlock_ratio

        return vec

    @staticmethod
    def _arrival_ratio(env: Any) -> float:
        n_agents = max(1, env.get_num_agents())
        arrived = sum(1 for agent in env.agents if str(agent.state).endswith("DONE"))
        return float(arrived) / float(n_agents)

    @staticmethod
    def _deadlock_ratio(env: Any) -> float:
        n_agents = max(1, env.get_num_agents())
        deadlocked = len(getattr(env.motionCheck, "svDeadlocked", set()))
        return float(deadlocked) / float(n_agents)

    def _compute_valid_actions(self, env: Any, handle: int) -> np.ndarray:
        # [do_nothing, move_left, move_forward, move_right, stop]
        valid = np.zeros((5,), dtype=np.bool_)
        valid[0] = True
        valid[4] = True

        agent = env.agents[handle]
        if agent.state.is_off_map_state():
            if env._elapsed_steps < agent.earliest_departure:
                return valid
            position = agent.initial_position
            direction = agent.initial_direction
        elif agent.state.is_on_map_state():
            position = agent.position
            direction = agent.direction
        else:
            return valid

        if position is None or direction is None:
            return valid

        transitions = env.rail.get_transitions(*position, direction)
        if transitions is None:
            return valid

        transitions = np.asarray(transitions)
        n_transitions = int(np.count_nonzero(transitions))
        orientation = int(direction)
        if n_transitions == 1:
            orientation = int(np.argmax(transitions))

        left = (orientation - 1) % 4
        forward = orientation % 4
        right = (orientation + 1) % 4

        valid[1] = bool(transitions[left])
        valid[2] = bool(transitions[forward])
        valid[3] = bool(transitions[right])
        return valid
