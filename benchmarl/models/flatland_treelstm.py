from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional

import torch
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.flatland_tree_modules import TreeLSTM, TreeTransformer


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
        use_tree_transformer: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.use_tree_transformer = use_tree_transformer

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

        if self.use_tree_transformer:
            self.tree_encoder = TreeTransformer(
                self.node_attr_size,
                self.tree_embedding_size,
                self.tree_embedding_size,
                n_nodes=self.num_nodes,
            )
        else:
            self.tree_encoder = TreeLSTM(
                self.node_attr_size,
                self.tree_embedding_size,
            )

        transformer_dim = self.hidden_size + self.tree_embedding_size
        transformer_layers = []
        for _ in range(self.transformer_layers):
            transformer_layers.append(
                nn.MultiheadAttention(
                    embed_dim=transformer_dim,
                    num_heads=self.transformer_heads,
                    batch_first=True,
                )
            )
            transformer_layers.append(
                nn.Sequential(
                    nn.Linear(transformer_dim, transformer_dim * self.transformer_ff_mult),
                    nn.GELU(),
                    nn.Linear(transformer_dim * self.transformer_ff_mult, transformer_dim),
                )
            )
        self.transformer_layers = nn.ModuleList(transformer_layers)
        self.att_mlp = nn.Sequential(
            nn.Linear(transformer_dim * 2, transformer_dim),
            nn.GELU(),
        )

    def _modify_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.clone()
        batch_size, n_agents, num_edges, _ = adjacency.shape
        num_nodes = num_edges + 1
        id_tree = torch.arange(0, batch_size * n_agents, device=adjacency.device)
        id_nodes = id_tree.view(batch_size, n_agents, 1)
        invalid_mask = adjacency == -2

        adjacency = adjacency.clone()
        adjacency[..., 0] += id_nodes * num_nodes
        adjacency[..., 1] += id_nodes * num_nodes

        fill_value = -batch_size * n_agents * num_nodes
        adjacency = torch.where(invalid_mask, adjacency.new_full((), fill_value), adjacency)
        adjacency = torch.where(adjacency < 0, adjacency.new_full((), -2), adjacency)
        return adjacency

    def _compute_embeddings(self, obs: TensorDictBase) -> torch.Tensor:
        agents_attr = obs.get(("agents", "observation", "agents_attr"))
        node_attr = obs.get(("agents", "observation", "node_attr"))
        adjacency = obs.get(("agents", "observation", "adjacency"))
        node_order = obs.get(("agents", "observation", "node_order"))
        edge_order = obs.get(("agents", "observation", "edge_order"))

        *batch_dims, n_agents, num_nodes, _ = node_attr.shape
        flat_batch = int(torch.tensor(batch_dims).prod()) if batch_dims else 1

        node_attr = node_attr.reshape(flat_batch, n_agents, num_nodes, -1)
        adjacency = adjacency.reshape(
            flat_batch, n_agents, adjacency.shape[-2], adjacency.shape[-1]
        )
        node_order = node_order.reshape(flat_batch, n_agents, -1)
        edge_order = edge_order.reshape(flat_batch, n_agents, -1)
        agents_attr = agents_attr.reshape(flat_batch, n_agents, -1)

        edge_count = edge_order.shape[-1]
        if adjacency.shape[-2] != edge_count:
            adjacency = adjacency[..., :edge_count, :]
        if adjacency.shape[-1] > 2:
            adjacency = adjacency[..., :2]

        adjacency = self._modify_adjacency(adjacency)
        if self.use_tree_transformer:
            tree_embedding = self.tree_encoder(node_attr, adjacency)
        else:
            tree_embedding = self.tree_encoder(node_attr, adjacency, node_order, edge_order)
            tree_embedding = tree_embedding.unflatten(0, (flat_batch, n_agents, num_nodes))
            tree_embedding = tree_embedding[:, :, 0, :]

        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=-1)

        att_embedding = embedding
        for i in range(0, len(self.transformer_layers), 2):
            att_layer = self.transformer_layers[i]
            ff_layer = self.transformer_layers[i + 1]
            att_out, _ = att_layer(att_embedding, att_embedding, att_embedding)
            att_embedding = self.att_mlp(torch.cat([att_embedding, att_out], dim=-1))
            att_embedding = ff_layer(att_embedding)

        if batch_dims:
            embedding = embedding.reshape(*batch_dims, n_agents, -1)
            att_embedding = att_embedding.reshape(*batch_dims, n_agents, -1)

        return embedding, att_embedding


class FlatlandTreePolicy(FlatlandTreeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + self.tree_embedding_size * 2, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self._compute_embeddings(tensordict)
        logits = self.policy_head(torch.cat([embedding, att_embedding], dim=-1))
        valid_actions = tensordict.get(("agents", "observation", "valid_actions"))
        logits = logits.masked_fill(~valid_actions, float("-inf"))
        tensordict.set(self.out_key, logits)
        return tensordict


class FlatlandTreeCritic(FlatlandTreeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        output_dim = self.output_leaf_spec.shape[-1]
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + self.tree_embedding_size * 2, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        embedding, att_embedding = self._compute_embeddings(tensordict)
        values = self.value_head(torch.cat([embedding, att_embedding], dim=-1))
        if not self.output_has_agent_dim:
            values = values.mean(-2)
        tensordict.set(self.out_key, values)
        return tensordict


@dataclass
class FlatlandTreePolicyConfig(ModelConfig):
    hidden_size: int = 128
    tree_embedding_size: int = 128
    num_nodes: int = 31
    num_edges: int = 30
    agent_attr_size: int = 83
    node_attr_size: int = 12
    num_actions: int = 5
    transformer_heads: int = 4
    transformer_layers: int = 3
    transformer_ff_mult: int = 2
    use_tree_transformer: bool = False

    @staticmethod
    def associated_class():
        return FlatlandTreePolicy


@dataclass
class FlatlandTreeCriticConfig(ModelConfig):
    hidden_size: int = 128
    tree_embedding_size: int = 128
    num_nodes: int = 31
    num_edges: int = 30
    agent_attr_size: int = 83
    node_attr_size: int = 12
    num_actions: int = 5
    transformer_heads: int = 4
    transformer_layers: int = 3
    transformer_ff_mult: int = 2
    use_tree_transformer: bool = False

    @staticmethod
    def associated_class():
        return FlatlandTreeCritic
