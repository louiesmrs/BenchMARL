from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional
import math

import torch
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.flatland_tree_modules import TreeLSTM, TreeTransformer


class AgentAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_mult: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ff_mult, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_out, _ = self.attn(x, x, x)
        x = self.att_mlp(torch.cat([x, att_out], dim=-1))
        x = self.ff(x)
        return x


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
        self.tree_encoder_type = tree_encoder_type
        self.agent_group = agent_group
        self.n_agents = n_agents

        self.embedding_key = (agent_group, "tree_embedding")
        self.att_embedding_key = (agent_group, "tree_att_embedding")
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
        cached = self._get_cached_embeddings(tensordict)
        if cached is not None:
            return cached
        embedding, att_embedding = self._compute_embeddings(tensordict)
        tensordict.set(self.embedding_key, embedding)
        tensordict.set(self.att_embedding_key, att_embedding)
        return embedding, att_embedding

    def _get_cached_embeddings(
        self, tensordict: TensorDictBase
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        keys = tensordict.keys(True, True)
        if self.embedding_key not in keys or self.att_embedding_key not in keys:
            return None
        agents_attr = tensordict.get(
            (self.agent_group, "observation", "agents_attr")
        )
        expected_shape = self._expected_embedding_shape(agents_attr)
        embedding = tensordict.get(self.embedding_key)
        att_embedding = tensordict.get(self.att_embedding_key)
        if embedding.shape != expected_shape or att_embedding.shape != expected_shape:
            return None
        return embedding, att_embedding

    def _expected_embedding_shape(self, agents_attr: torch.Tensor) -> tuple[int, ...]:
        batch_dims = agents_attr.shape[:-2]
        if batch_dims:
            return (*batch_dims, self.n_agents, self.embedding_dim)
        return (self.n_agents, self.embedding_dim)

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

    def _compute_embeddings(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_prefix = (self.agent_group, "observation")
        agents_attr = tensordict.get((*obs_prefix, "agents_attr"))
        node_attr = tensordict.get((*obs_prefix, "node_attr"))
        adjacency = tensordict.get((*obs_prefix, "adjacency"))
        node_order = tensordict.get((*obs_prefix, "node_order"))
        edge_order = tensordict.get((*obs_prefix, "edge_order"))

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

        adjacency = self._modify_adjacency(adjacency)

        if self.tree_encoder_type == "lstm":
            tree_embedding = self.tree_encoder(
                node_attr, adjacency, node_order, edge_order
            )
            tree_embedding = tree_embedding.reshape(
                flat_batch, self.n_agents, self.num_nodes, -1
            )[:, :, 0, :]
        else:
            tree_embedding = self.tree_encoder(node_attr, adjacency, node_order)

        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=-1)

        att_embedding = embedding
        for layer in self.attention_layers:
            att_embedding = layer(att_embedding)

        if batch_dims:
            embedding = embedding.reshape(*batch_dims, self.n_agents, -1)
            att_embedding = att_embedding.reshape(*batch_dims, self.n_agents, -1)

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
