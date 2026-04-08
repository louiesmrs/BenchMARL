from __future__ import annotations

import math

import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite, UnboundedContinuous, UnboundedDiscrete
from torchrl.envs.transforms import Transform


class FlatlandTreePreprocessor(Transform):
    def __init__(self, agent_group: str = "agents", branching_factor: int = 3) -> None:
        super().__init__()
        self.agent_group = agent_group
        self.branching_factor = branching_factor

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_key = (self.agent_group, "observation")
        adjacency_key = (*obs_key, "adjacency")
        node_order_key = (*obs_key, "node_order")
        edge_order_key = (*obs_key, "edge_order")
        keys = tensordict.keys(True, True)
        if adjacency_key not in keys or node_order_key not in keys or edge_order_key not in keys:
            raise KeyError(
                "FlatlandTreePreprocessor missing required keys "
                f"(adjacency/node_order/edge_order) in {keys}"
            )

        adjacency = tensordict.get(adjacency_key)
        node_order = tensordict.get(node_order_key)
        edge_order = tensordict.get(edge_order_key)

        adjacency, num_edges, n_agents, batch_dims = self._reshape_adjacency(adjacency)
        node_order = self._reshape_with_agents(node_order, n_agents, batch_dims)
        edge_order = self._reshape_with_agents(edge_order, n_agents, batch_dims)

        num_nodes = node_order.shape[-1]
        adjacency_offset = self._offset_adjacency(adjacency, num_nodes)
        positional_encoding = self._positional_encoding(
            adjacency_offset, node_order, num_nodes
        )

        tensordict.set(
            (*obs_key, "adjacency_offset"),
            self._restore_batch(adjacency_offset, batch_dims),
        )
        tensordict.set(
            (*obs_key, "positional_encoding"),
            self._restore_batch(positional_encoding, batch_dims),
        )
        return tensordict

    def transform_observation_spec(self, observation_spec):
        if not isinstance(observation_spec, Composite):
            return observation_spec
        if self.agent_group not in observation_spec.keys():
            return observation_spec
        obs_group = observation_spec[self.agent_group]
        if "observation" not in obs_group.keys():
            return observation_spec
        obs_spec = obs_group["observation"]
        if "adjacency" not in obs_spec.keys():
            return observation_spec
        if "node_order" not in obs_spec.keys():
            return observation_spec

        adjacency_spec = obs_spec["adjacency"]
        if adjacency_spec.shape[-1] == 3:
            num_edges = adjacency_spec.shape[-2]
            adjacency_shape = adjacency_spec.shape
        else:
            if adjacency_spec.shape[-1] % 3 != 0:
                return observation_spec
            num_edges = adjacency_spec.shape[-1] // 3
            adjacency_shape = (*adjacency_spec.shape[:-1], num_edges, 3)
        num_nodes = num_edges + 1
        positional_dim = num_nodes * self.branching_factor

        spec_device = adjacency_spec.device
        obs_spec["adjacency_offset"] = UnboundedDiscrete(
            shape=adjacency_shape,
            dtype=torch.int64,
            device=spec_device,
        )
        obs_spec["positional_encoding"] = UnboundedContinuous(
            shape=(*obs_spec.shape, num_nodes, positional_dim),
            dtype=torch.float32,
            device=spec_device,
        )
        return observation_spec

    def _reshape_adjacency(self, adjacency: torch.Tensor):
        if adjacency.shape[-1] == 3:
            batch_dims = adjacency.shape[:-3]
            n_agents = adjacency.shape[-3]
            num_edges = adjacency.shape[-2]
        else:
            if adjacency.shape[-1] % 3 != 0:
                raise ValueError("Flatland adjacency must have last dim divisible by 3")
            num_edges = adjacency.shape[-1] // 3
            batch_dims = adjacency.shape[:-2]
            n_agents = adjacency.shape[-2]
            adjacency = adjacency.reshape(*batch_dims, n_agents, num_edges, 3)
        flat_batch = int(math.prod(batch_dims)) if batch_dims else 1
        adjacency = adjacency.reshape(flat_batch, n_agents, num_edges, 3)
        return adjacency, num_edges, n_agents, batch_dims

    def _reshape_with_agents(
        self, tensor: torch.Tensor, n_agents: int, batch_dims: tuple[int, ...]
    ):
        flat_batch = int(math.prod(batch_dims)) if batch_dims else 1
        return tensor.reshape(flat_batch, n_agents, tensor.shape[-1])

    def _restore_batch(self, tensor: torch.Tensor, batch_dims: tuple[int, ...]):
        if batch_dims:
            return tensor.reshape(*batch_dims, *tensor.shape[1:])
        return tensor.squeeze(0)

    def _offset_adjacency(self, adjacency: torch.Tensor, num_nodes: int):
        batch_size, n_agents, num_edges, num_cols = adjacency.shape
        idx = adjacency[..., :2]
        invalid = idx < 0
        offset = (
            torch.arange(batch_size * n_agents, device=adjacency.device)
            .view(batch_size, n_agents, 1, 1)
            .mul(num_nodes)
        )
        idx = idx + offset
        idx = torch.where(invalid, idx.new_full((), -1), idx)

        if num_cols < 3:
            return idx
        slot = adjacency[..., 2:3]
        invalid_edge = invalid.any(-1, keepdim=True)
        slot = torch.where(invalid_edge, slot.new_full((), -1), slot)
        return torch.cat([idx, slot], dim=-1)

    def _positional_encoding(
        self, adjacency: torch.Tensor, node_order: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        batch_size, n_agents, num_edges, _ = adjacency.shape
        tree_count = batch_size * n_agents
        positional_dim = num_nodes * self.branching_factor

        node_order = node_order.reshape(tree_count, num_nodes)
        max_order = node_order.max(dim=1).values
        node_depth = (max_order.unsqueeze(1) - node_order).to(torch.long)

        pos_flat = torch.zeros(
            tree_count * num_nodes, positional_dim, device=adjacency.device
        )

        edges = adjacency.reshape(tree_count, num_edges, -1).reshape(-1, adjacency.shape[-1])
        parent_idx = edges[:, 0].to(torch.long)
        child_idx = edges[:, 1].to(torch.long)
        slot = edges[:, 2].to(torch.long) if edges.shape[1] > 2 else None

        valid_edge = (parent_idx >= 0) & (child_idx >= 0)
        safe_parent = parent_idx.clamp(0, pos_flat.shape[0] - 1)
        safe_child = child_idx.clamp(0, pos_flat.shape[0] - 1)

        if slot is None:
            slot_idx = torch.zeros_like(parent_idx)
        else:
            slot_idx = (slot + 1).clamp(0, self.branching_factor - 1)

        node_depth_flat = node_depth.reshape(-1)
        parent_depth = node_depth_flat[safe_parent]

        for depth in range(num_nodes):
            depth_mask = (parent_depth == depth) & valid_edge
            edge_weight = depth_mask.to(pos_flat.dtype).unsqueeze(-1)
            parent_pos = pos_flat[safe_parent]
            index = depth * self.branching_factor + slot_idx
            onehot = torch.zeros_like(parent_pos).scatter(1, index.unsqueeze(1), 1.0)
            child_pos = (parent_pos + onehot) * edge_weight
            pos_flat.index_add_(0, safe_child, child_pos)

        pos = pos_flat.view(tree_count, num_nodes, positional_dim)
        return pos.reshape(batch_size, n_agents, num_nodes, positional_dim)
