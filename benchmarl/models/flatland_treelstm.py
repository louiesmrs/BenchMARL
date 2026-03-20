from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, MISSING
from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.flatland_tree_modules import (
    AgentAttentionBlock,
    TreeLSTM,
    TreeTransformer,
)

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    from torch_geometric.nn import GATv2Conv


class FlatlandTreeEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        tree_embedding_size: int,
        num_nodes: int,
        num_edges: int,
        agent_attr_size: int,
        node_attr_size: int,
        transformer_heads: int,
        transformer_layers: int,
        transformer_ff_mult: int,
        tree_encoder_type: str,
        agent_group: str,
        n_agents: int,
        gnn_heads: int = 4,
        gnn_layers: int = 2,
        gnn_dropout: float = 0.0,
        gnn_self_loops: bool = False,
        gnn_add_reverse_edges: bool = True,
        gnn_use_edge_attr: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tree_embedding_size = tree_embedding_size
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.agent_attr_size = agent_attr_size
        self.node_attr_size = node_attr_size
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_ff_mult = transformer_ff_mult
        self.gnn_heads = gnn_heads
        self.gnn_layers = gnn_layers
        self.gnn_dropout = gnn_dropout
        self.gnn_self_loops = gnn_self_loops
        self.gnn_add_reverse_edges = gnn_add_reverse_edges
        self.gnn_use_edge_attr = gnn_use_edge_attr
        self.branching_factor = 3
        self.tree_encoder_type = tree_encoder_type
        self.agent_group = agent_group
        self.n_agents = n_agents

        self.embedding_dim = self.hidden_size + self.tree_embedding_size

        self.attr_embedding = nn.Sequential(
            nn.Linear(self.agent_attr_size, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
        )

        if self.tree_encoder_type == "lstm":
            self.tree_encoder = TreeLSTM(
                self.node_attr_size,
                self.tree_embedding_size,
                max_iterations=self.num_nodes,
            )
        elif self.tree_encoder_type == "transformer":
            self.tree_encoder = TreeTransformer(
                self.node_attr_size,
                self.tree_embedding_size,
                self.tree_embedding_size,
                n_nodes=self.num_nodes,
                num_heads=self.transformer_heads,
                num_layers=max(1, self.transformer_layers),
                ff_mult=self.transformer_ff_mult,
            )
        elif self.tree_encoder_type == "gnn":
            if not _has_torch_geometric:
                raise ImportError(
                    "FlatlandTreeGNN requires torch_geometric but it is not installed."
                )
            if self.gnn_layers < 1:
                raise ValueError("gnn_layers must be >= 1")
            edge_dim = self.branching_factor if self.gnn_use_edge_attr else None
            gnn_layers = []
            in_channels = self.node_attr_size
            for _ in range(self.gnn_layers):
                gnn_layers.append(
                    GATv2Conv(
                        in_channels=in_channels,
                        out_channels=self.tree_embedding_size,
                        heads=self.gnn_heads,
                        concat=False,
                        dropout=self.gnn_dropout,
                        add_self_loops=False,
                        edge_dim=edge_dim,
                    )
                )
                in_channels = self.tree_embedding_size
            self.tree_encoder = nn.ModuleList(gnn_layers)
            self.gnn_activation = nn.GELU()
        else:
            raise ValueError(f"Unknown tree encoder type: {self.tree_encoder_type}")

        transformer_dim = self.embedding_dim
        self.attention_layers = nn.ModuleList(
            [
                AgentAttentionBlock(
                    transformer_dim, self.transformer_heads, self.transformer_ff_mult
                )
                for _ in range(self.transformer_layers)
            ]
        )

    def forward(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._compute_embeddings(tensordict)

    def _modify_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.clone()
        batch_size, n_agents, num_edges, num_cols = adjacency.shape
        num_nodes = num_edges + 1

        idx = adjacency[..., :2]
        invalid = idx < 0
        offset = (
            torch.arange(batch_size * n_agents, device=adjacency.device)
            .view(batch_size, n_agents, 1, 1)
            .mul(num_nodes)
        )
        idx = idx + offset
        idx = torch.where(invalid, idx.new_full((), -1), idx)

        if num_cols > 2:
            slot = adjacency[..., 2:3]
            invalid_edge = invalid.any(-1, keepdim=True)
            slot = torch.where(invalid_edge, slot.new_full((), -1), slot)
            return torch.cat([idx, slot], dim=-1)
        return idx

    def _build_tree_gnn_edges(
        self, adjacency: torch.Tensor, node_features: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        edge_idx = adjacency[..., :2].reshape(-1, 2)
        valid = (edge_idx >= 0).all(-1)
        torch._assert(valid.any(), "TreeGNN received no valid edges.")

        edge_idx = edge_idx[valid].to(torch.long)
        edge_index = edge_idx.t().contiguous()
        edge_attr = None

        if self.gnn_use_edge_attr:
            if adjacency.shape[-1] > 2:
                slot = adjacency[..., 2].reshape(-1)[valid].to(torch.long)
                slot = slot.clamp(min=0, max=self.branching_factor - 1)
                edge_attr = F.one_hot(
                    slot, num_classes=self.branching_factor
                ).to(node_features.dtype)
            else:
                edge_attr = torch.zeros(
                    edge_index.shape[1],
                    self.branching_factor,
                    device=node_features.device,
                    dtype=node_features.dtype,
                )

        if self.gnn_add_reverse_edges:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        if self.gnn_self_loops:
            num_nodes = node_features.shape[0]
            self_loops = torch.arange(num_nodes, device=node_features.device)
            self_index = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([edge_index, self_index], dim=1)
            if edge_attr is not None:
                self_attr = torch.zeros(
                    num_nodes,
                    edge_attr.shape[-1],
                    device=node_features.device,
                    dtype=node_features.dtype,
                )
                edge_attr = torch.cat([edge_attr, self_attr], dim=0)

        return edge_index, edge_attr

    def _compute_embeddings(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_prefix = (self.agent_group, "observation")
        keys = tensordict.keys(True, True)
        agents_attr = tensordict.get((*obs_prefix, "agents_attr"))
        node_attr = tensordict.get((*obs_prefix, "node_attr"))
        adjacency = tensordict.get((*obs_prefix, "adjacency"))
        node_order = tensordict.get((*obs_prefix, "node_order"))
        edge_order = tensordict.get((*obs_prefix, "edge_order"))

        adjacency_offset_key = (*obs_prefix, "adjacency_offset")
        positional_encoding_key = (*obs_prefix, "positional_encoding")
        has_adjacency_offset = adjacency_offset_key in keys
        if has_adjacency_offset:
            adjacency = tensordict.get(adjacency_offset_key)

        positional_encoding = (
            tensordict.get(positional_encoding_key)
            if positional_encoding_key in keys
            else None
        )

        batch_dims = agents_attr.shape[:-2]
        flat_batch = int(math.prod(batch_dims)) if batch_dims else 1

        agents_attr = agents_attr.reshape(flat_batch, self.n_agents, -1)

        if node_attr.dim() == 3:
            node_attr = node_attr.reshape(
                flat_batch, self.n_agents, self.num_nodes, self.node_attr_size
            )
        else:
            node_attr = node_attr.reshape(
                flat_batch, self.n_agents, self.num_nodes, -1
            )

        if adjacency.dim() == 3:
            adjacency = adjacency.reshape(flat_batch, self.n_agents, self.num_edges, 3)
        else:
            adjacency = adjacency.reshape(
                flat_batch, self.n_agents, self.num_edges, -1
            )

        node_order = node_order.reshape(flat_batch, self.n_agents, self.num_nodes)
        edge_order = edge_order.reshape(flat_batch, self.n_agents, self.num_edges)

        if not has_adjacency_offset and not (
            positional_encoding is not None and self.tree_encoder_type == "transformer"
        ):
            adjacency = self._modify_adjacency(adjacency)

        if positional_encoding is not None:
            positional_encoding = positional_encoding.reshape(
                flat_batch, self.n_agents, self.num_nodes, -1
            )

        if self.tree_encoder_type == "lstm":
            tree_embedding = self.tree_encoder(
                node_attr, adjacency, node_order, edge_order
            )
            tree_embedding = tree_embedding.reshape(
                flat_batch, self.n_agents, self.num_nodes, -1
            )[:, :, 0, :]
        elif self.tree_encoder_type == "transformer":
            tree_embedding = self.tree_encoder(
                node_attr,
                adjacency,
                node_order,
                positional_encoding=positional_encoding,
            )
        elif self.tree_encoder_type == "gnn":
            node_features = node_attr.reshape(
                flat_batch * self.n_agents * self.num_nodes, -1
            )
            edge_index, edge_attr = self._build_tree_gnn_edges(
                adjacency, node_features
            )
            gnn_out = node_features
            for layer_idx, gnn_layer in enumerate(self.tree_encoder):
                if edge_attr is None:
                    gnn_out = gnn_layer(gnn_out, edge_index)
                else:
                    gnn_out = gnn_layer(gnn_out, edge_index, edge_attr)
                if layer_idx < len(self.tree_encoder) - 1:
                    gnn_out = self.gnn_activation(gnn_out)
            gnn_out = gnn_out.reshape(
                flat_batch, self.n_agents, self.num_nodes, -1
            )
            tree_embedding = gnn_out[:, :, 0, :]
        else:
            raise ValueError(f"Unknown tree encoder type: {self.tree_encoder_type}")

        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=-1)

        att_embedding = embedding
        for layer in self.attention_layers:
            att_embedding = layer(att_embedding)

        if batch_dims:
            embedding = embedding.reshape(*batch_dims, self.n_agents, -1)
            att_embedding = att_embedding.reshape(*batch_dims, self.n_agents, -1)
        else:
            embedding = embedding.squeeze(0)
            att_embedding = att_embedding.squeeze(0)

        return embedding, att_embedding


class FlatlandTreeBase(Model):
    def __init__(
        self,
        hidden_size: int,
        tree_embedding_size: int,
        num_nodes: int,
        num_edges: int,
        agent_attr_size: int,
        node_attr_size: int,
        num_actions: int,
        transformer_heads: int,
        transformer_layers: int,
        transformer_ff_mult: int,
        tree_encoder_type: str,
        gnn_heads: int = 4,
        gnn_layers: int = 2,
        gnn_dropout: float = 0.0,
        gnn_self_loops: bool = False,
        gnn_add_reverse_edges: bool = True,
        gnn_use_edge_attr: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.tree_embedding_size = tree_embedding_size
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.agent_attr_size = agent_attr_size
        self.node_attr_size = node_attr_size
        self.num_actions = num_actions
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_ff_mult = transformer_ff_mult
        self.gnn_heads = gnn_heads
        self.gnn_layers = gnn_layers
        self.gnn_dropout = gnn_dropout
        self.gnn_self_loops = gnn_self_loops
        self.gnn_add_reverse_edges = gnn_add_reverse_edges
        self.gnn_use_edge_attr = gnn_use_edge_attr
        self.tree_encoder_type = tree_encoder_type
        self.embedding_dim = self.hidden_size + self.tree_embedding_size
        self.head_input_dim = self.embedding_dim * 2

        super().__init__(**kwargs)

        if not self.input_has_agent_dim:
            raise ValueError("Flatland tree models require per-agent observations.")

        self.encoder = FlatlandTreeEncoder(
            hidden_size=self.hidden_size,
            tree_embedding_size=self.tree_embedding_size,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            agent_attr_size=self.agent_attr_size,
            node_attr_size=self.node_attr_size,
            transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers,
            transformer_ff_mult=self.transformer_ff_mult,
            tree_encoder_type=self.tree_encoder_type,
            agent_group=self.agent_group,
            n_agents=self.n_agents,
            gnn_heads=self.gnn_heads,
            gnn_layers=self.gnn_layers,
            gnn_dropout=self.gnn_dropout,
            gnn_self_loops=self.gnn_self_loops,
            gnn_add_reverse_edges=self.gnn_add_reverse_edges,
            gnn_use_edge_attr=self.gnn_use_edge_attr,
        )

        self.to(self.device)

    def _perform_checks(self):
        super()._perform_checks()
        if self.agent_group not in self.input_spec.keys():
            raise ValueError(
                f"Expected agent group '{self.agent_group}' in input spec, got {self.input_spec}"
            )
        obs_spec = self.input_spec[self.agent_group]
        required_keys = {
            "agents_attr",
            "node_attr",
            "adjacency",
            "node_order",
            "edge_order",
            "valid_actions",
        }
        for key in required_keys:
            if (
                self.agent_group,
                "observation",
                key,
            ) not in self.input_spec.keys(True, True):
                raise ValueError(
                    f"Missing observation key '{key}' in input spec {self.input_spec}"
                )

        agents_attr_spec = obs_spec["observation"]["agents_attr"]
        node_attr_spec = obs_spec["observation"]["node_attr"]
        adjacency_spec = obs_spec["observation"]["adjacency"]
        node_order_spec = obs_spec["observation"]["node_order"]
        edge_order_spec = obs_spec["observation"]["edge_order"]

        if agents_attr_spec.shape[-1] != self.agent_attr_size:
            raise ValueError(
                "agents_attr last dim "
                f"{agents_attr_spec.shape[-1]} does not match agent_attr_size {self.agent_attr_size}"
            )

        if len(node_attr_spec.shape) == 2:
            expected = self.num_nodes * self.node_attr_size
            if node_attr_spec.shape[-1] != expected:
                raise ValueError(
                    f"node_attr flattened shape {node_attr_spec.shape} does not match expected {expected}"
                )
        else:
            if (
                node_attr_spec.shape[-2] != self.num_nodes
                or node_attr_spec.shape[-1] != self.node_attr_size
            ):
                raise ValueError(
                    f"node_attr shape {node_attr_spec.shape} does not match num_nodes {self.num_nodes} and node_attr_size {self.node_attr_size}"
                )

        if len(adjacency_spec.shape) == 2:
            expected = self.num_edges * 3
            if adjacency_spec.shape[-1] != expected:
                raise ValueError(
                    f"adjacency flattened shape {adjacency_spec.shape} does not match expected {expected}"
                )
        else:
            if adjacency_spec.shape[-2] != self.num_edges:
                raise ValueError(
                    f"adjacency edges {adjacency_spec.shape[-2]} does not match num_edges {self.num_edges}"
                )
            if adjacency_spec.shape[-1] < 3:
                raise ValueError("adjacency must have three columns (parent, child, slot)")

        if node_order_spec.shape[-1] != self.num_nodes:
            raise ValueError(
                f"node_order length {node_order_spec.shape[-1]} does not match num_nodes {self.num_nodes}"
            )
        if edge_order_spec.shape[-1] != self.num_edges:
            raise ValueError(
                f"edge_order length {edge_order_spec.shape[-1]} does not match num_edges {self.num_edges}"
            )


class FlatlandTreeLSTMFeature(FlatlandTreeBase):
    """Tree-based Flatland feature extractor.

    This module consumes the native Flatland tree observation and writes a learned
    feature tensor to ``self.out_key``. It does not produce policy logits or values,
    and it does not apply action masking.
    """

    def __init__(self, **kwargs):
        super().__init__(tree_encoder_type="lstm", **kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        self.feature_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )
        self.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self.encoder(tensordict)
        features = self.feature_head(torch.cat([embedding, att_embedding], dim=-1))
        if not self.output_has_agent_dim:
            features = features.mean(dim=-2)
        tensordict.set(self.out_key, features)
        return tensordict


class FlatlandTreeLSTMPolicy(FlatlandTreeBase):
    def __init__(self, **kwargs):
        super().__init__(tree_encoder_type="lstm", **kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        if output_dim != self.num_actions:
            raise ValueError(
                f"Policy output dim {output_dim} does not match num_actions {self.num_actions}"
            )
        self.policy_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )
        self.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self.encoder(tensordict)
        logits = self.policy_head(torch.cat([embedding, att_embedding], dim=-1))
        valid_actions = tensordict.get(
            (self.agent_group, "observation", "valid_actions")
        )
        logits = logits.masked_fill(~valid_actions, float("-inf"))
        tensordict.set(self.out_key, logits)
        return tensordict


class FlatlandTreeTransformerPolicy(FlatlandTreeBase):
    def __init__(self, **kwargs):
        super().__init__(tree_encoder_type="transformer", **kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        if output_dim != self.num_actions:
            raise ValueError(
                f"Policy output dim {output_dim} does not match num_actions {self.num_actions}"
            )
        self.policy_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )
        self.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self.encoder(tensordict)
        logits = self.policy_head(torch.cat([embedding, att_embedding], dim=-1))
        valid_actions = tensordict.get(
            (self.agent_group, "observation", "valid_actions")
        )
        logits = logits.masked_fill(~valid_actions, float("-inf"))
        tensordict.set(self.out_key, logits)
        return tensordict


class FlatlandTreeGNNPolicy(FlatlandTreeBase):
    def __init__(self, **kwargs):
        super().__init__(tree_encoder_type="gnn", **kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        if output_dim != self.num_actions:
            raise ValueError(
                f"Policy output dim {output_dim} does not match num_actions {self.num_actions}"
            )
        self.policy_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )
        self.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self.encoder(tensordict)
        logits = self.policy_head(torch.cat([embedding, att_embedding], dim=-1))
        valid_actions = tensordict.get(
            (self.agent_group, "observation", "valid_actions")
        )
        logits = logits.masked_fill(~valid_actions, float("-inf"))
        tensordict.set(self.out_key, logits)
        return tensordict


class FlatlandTreeLSTMCritic(FlatlandTreeBase):
    def __init__(self, **kwargs):
        super().__init__(tree_encoder_type="lstm", **kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        self.value_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )
        self.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self.encoder(tensordict)
        values = self.value_head(torch.cat([embedding, att_embedding], dim=-1))
        if not self.output_has_agent_dim:
            values = values.mean(-2)
        tensordict.set(self.out_key, values)
        return tensordict


class FlatlandTreeTransformerCritic(FlatlandTreeBase):
    def __init__(self, **kwargs):
        super().__init__(tree_encoder_type="transformer", **kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        self.value_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )
        self.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self.encoder(tensordict)
        values = self.value_head(torch.cat([embedding, att_embedding], dim=-1))
        if not self.output_has_agent_dim:
            values = values.mean(-2)
        tensordict.set(self.out_key, values)
        return tensordict


class FlatlandTreeGNNCritic(FlatlandTreeBase):
    def __init__(self, **kwargs):
        super().__init__(tree_encoder_type="gnn", **kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        self.value_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )
        self.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self.encoder(tensordict)
        values = self.value_head(torch.cat([embedding, att_embedding], dim=-1))
        if not self.output_has_agent_dim:
            values = values.mean(-2)
        tensordict.set(self.out_key, values)
        return tensordict


@dataclass
class FlatlandTreeLSTMFeatureConfig(ModelConfig):
    hidden_size: int = MISSING
    tree_embedding_size: int = MISSING
    num_nodes: int = MISSING
    num_edges: int = MISSING
    agent_attr_size: int = MISSING
    node_attr_size: int = MISSING
    num_actions: int = MISSING
    transformer_heads: int = MISSING
    transformer_layers: int = MISSING
    transformer_ff_mult: int = MISSING

    @staticmethod
    def associated_class():
        return FlatlandTreeLSTMFeature


@dataclass
class FlatlandTreeLSTMPolicyConfig(ModelConfig):
    hidden_size: int = MISSING
    tree_embedding_size: int = MISSING
    num_nodes: int = MISSING
    num_edges: int = MISSING
    agent_attr_size: int = MISSING
    node_attr_size: int = MISSING
    num_actions: int = MISSING
    transformer_heads: int = MISSING
    transformer_layers: int = MISSING
    transformer_ff_mult: int = MISSING

    @staticmethod
    def associated_class():
        return FlatlandTreeLSTMPolicy


@dataclass
class FlatlandTreeTransformerPolicyConfig(ModelConfig):
    hidden_size: int = MISSING
    tree_embedding_size: int = MISSING
    num_nodes: int = MISSING
    num_edges: int = MISSING
    agent_attr_size: int = MISSING
    node_attr_size: int = MISSING
    num_actions: int = MISSING
    transformer_heads: int = MISSING
    transformer_layers: int = MISSING
    transformer_ff_mult: int = MISSING

    @staticmethod
    def associated_class():
        return FlatlandTreeTransformerPolicy


@dataclass
class FlatlandTreeGNNPolicyConfig(ModelConfig):
    hidden_size: int = MISSING
    tree_embedding_size: int = MISSING
    num_nodes: int = MISSING
    num_edges: int = MISSING
    agent_attr_size: int = MISSING
    node_attr_size: int = MISSING
    num_actions: int = MISSING
    transformer_heads: int = MISSING
    transformer_layers: int = MISSING
    transformer_ff_mult: int = MISSING
    gnn_heads: int = MISSING
    gnn_layers: int = MISSING
    gnn_dropout: float = MISSING
    gnn_self_loops: bool = MISSING
    gnn_add_reverse_edges: bool = MISSING
    gnn_use_edge_attr: bool = MISSING

    @staticmethod
    def associated_class():
        return FlatlandTreeGNNPolicy


@dataclass
class FlatlandTreeLSTMCriticConfig(ModelConfig):
    hidden_size: int = MISSING
    tree_embedding_size: int = MISSING
    num_nodes: int = MISSING
    num_edges: int = MISSING
    agent_attr_size: int = MISSING
    node_attr_size: int = MISSING
    num_actions: int = MISSING
    transformer_heads: int = MISSING
    transformer_layers: int = MISSING
    transformer_ff_mult: int = MISSING

    @staticmethod
    def associated_class():
        return FlatlandTreeLSTMCritic


@dataclass
class FlatlandTreeTransformerCriticConfig(ModelConfig):
    hidden_size: int = MISSING
    tree_embedding_size: int = MISSING
    num_nodes: int = MISSING
    num_edges: int = MISSING
    agent_attr_size: int = MISSING
    node_attr_size: int = MISSING
    num_actions: int = MISSING
    transformer_heads: int = MISSING
    transformer_layers: int = MISSING
    transformer_ff_mult: int = MISSING

    @staticmethod
    def associated_class():
        return FlatlandTreeTransformerCritic


@dataclass
class FlatlandTreeGNNCriticConfig(ModelConfig):
    hidden_size: int = MISSING
    tree_embedding_size: int = MISSING
    num_nodes: int = MISSING
    num_edges: int = MISSING
    agent_attr_size: int = MISSING
    node_attr_size: int = MISSING
    num_actions: int = MISSING
    transformer_heads: int = MISSING
    transformer_layers: int = MISSING
    transformer_ff_mult: int = MISSING
    gnn_heads: int = MISSING
    gnn_layers: int = MISSING
    gnn_dropout: float = MISSING
    gnn_self_loops: bool = MISSING
    gnn_add_reverse_edges: bool = MISSING
    gnn_use_edge_attr: bool = MISSING

    @staticmethod
    def associated_class():
        return FlatlandTreeGNNCritic
