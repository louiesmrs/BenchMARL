from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TreeLSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        branching_factor: int = 3,
        max_iterations: int = 31,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.branching_factor = branching_factor
        self.max_iterations = max_iterations

        self.W_iou = nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = nn.Linear(3 * self.out_features, 3 * self.out_features, bias=False)
        self.W_c = nn.Linear(3 * self.out_features, self.out_features)

        self.W_f = nn.Linear(self.in_features, self.out_features)
        self.U_f = nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(
        self,
        forest: torch.Tensor,
        adjacency: torch.Tensor,
        node_order: torch.Tensor,
        edge_order: torch.Tensor,
    ) -> torch.Tensor:
        if adjacency.shape[-1] < 3:
            raise ValueError(
                "TreeLSTM expects adjacency with three columns (parent, child, slot)"
            )

        forest_flat = forest.reshape(-1, self.in_features)
        adjacency_flat = adjacency[..., :3].reshape(-1, 3)
        node_order_flat = node_order.reshape(-1)
        edge_order_flat = edge_order.reshape(-1)

        total_nodes = forest_flat.shape[0]
        device = forest_flat.device

        h = torch.zeros(total_nodes, self.out_features, device=device)
        c = torch.zeros(total_nodes, self.out_features, device=device)

        if node_order_flat.numel() == 0:
            return h

        parent_idx = adjacency_flat[:, 0].to(torch.long)
        child_idx = adjacency_flat[:, 1].to(torch.long)
        slot_idx = (adjacency_flat[:, 2].to(torch.long) + 1).clamp(
            0, self.branching_factor - 1
        )

        valid_edge = (
            (parent_idx >= 0)
            & (parent_idx < total_nodes)
            & (child_idx >= 0)
            & (child_idx < total_nodes)
        )

        safe_parent = parent_idx.clamp(0, total_nodes - 1)
        safe_child = child_idx.clamp(0, total_nodes - 1)
        flat_index = safe_parent * self.branching_factor + slot_idx
        for iteration in range(self.max_iterations):
            node_mask = (node_order_flat == iteration).unsqueeze(-1)
            edge_mask = edge_order_flat == iteration
            edge_weight = (valid_edge & edge_mask).unsqueeze(-1).to(forest_flat.dtype)

            child_h = h[safe_child] * edge_weight
            child_c = c[safe_child] * edge_weight

            flat_h = torch.zeros(
                total_nodes * self.branching_factor,
                self.out_features,
                device=device,
            )
            flat_c = torch.zeros_like(flat_h)
            flat_h = flat_h.index_add(0, flat_index, child_h)
            flat_c = flat_c.index_add(0, flat_index, child_c)

            child_h_merge = flat_h.view(
                total_nodes, self.branching_factor * self.out_features
            )
            child_c_merge = flat_c.view(
                total_nodes, self.branching_factor * self.out_features
            )

            iou = self.W_iou(forest_flat) + self.U_iou(child_h_merge)
            i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            u = torch.tanh(u)

            parent_feat = forest_flat[safe_parent]
            f = self.W_f(parent_feat) + self.U_f(child_h)
            f = torch.sigmoid(f) * edge_weight
            fc = f * child_c

            flat_fc = torch.zeros_like(flat_h)
            flat_fc = flat_fc.index_add(0, flat_index, fc)
            c_reduce = self.W_c(
                flat_fc.view(total_nodes, self.branching_factor * self.out_features)
            )

            new_c = i * u + c_reduce
            new_h = o * torch.tanh(new_c)

            c = torch.where(node_mask, new_c, c)
            h = torch.where(node_mask, new_h, h)

        return h


class TreeTransformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_nodes: int,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 2,
        branching_factor: int = 3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_nodes = n_nodes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_mult = ff_mult
        self.branching_factor = branching_factor

        self.input_linear = nn.Linear(self.in_features, self.hidden_features)
        self.output_linear = nn.Linear(
            self.hidden_features * self.n_nodes, self.out_features
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_features,
            nhead=self.num_heads,
            batch_first=True,
            dim_feedforward=self.hidden_features * self.ff_mult,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        self.norm = nn.LayerNorm(self.out_features)

    def forward(
        self,
        forest: torch.Tensor,
        adjacency: torch.Tensor,
        node_order: torch.Tensor,
    ) -> torch.Tensor:
        if adjacency.shape[-1] < 2:
            raise ValueError(
                "TreeTransformer expects adjacency with at least two columns (parent, child)"
            )
        batch_size, n_agents, n_nodes, _ = forest.shape
        if n_nodes != self.n_nodes:
            raise ValueError(
                f"TreeTransformer expected {self.n_nodes} nodes but got {n_nodes}"
            )

        flat_batch = batch_size * n_agents
        forest = forest.reshape(flat_batch, n_nodes, self.in_features)
        node_order = node_order.reshape(flat_batch, n_nodes)
        adjacency = adjacency.reshape(flat_batch, adjacency.shape[-2], adjacency.shape[-1])

        positional_encoding = self.get_positional_encoding(adjacency, node_order)
        input_data = self.input_linear(forest) + positional_encoding
        output = self.transformer_encoder(input_data)
        output = output.reshape(flat_batch, n_nodes * self.hidden_features)
        output = self.output_linear(output).reshape(batch_size, n_agents, -1)
        return self.norm(output)

    def get_positional_encoding(
        self, adjacency: torch.Tensor, node_order: torch.Tensor
    ) -> torch.Tensor:
        device = adjacency.device
        tree_count, n_nodes = node_order.shape

        max_order = node_order.max(dim=1).values
        node_depth = (max_order.unsqueeze(1) - node_order).to(torch.long)
        max_depth = self.n_nodes - 1
        required_dim = (max_depth + 1) * self.branching_factor
        if required_dim > self.hidden_features:
            raise ValueError(
                "TreeTransformer positional encoding requires "
                f"hidden_features >= {required_dim}, got {self.hidden_features}"
            )

        pos_flat = torch.zeros(
            tree_count * n_nodes, self.hidden_features, device=device
        )

        edges = adjacency.reshape(-1, adjacency.shape[-1])
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

        for depth in range(max_depth + 1):
            depth_mask = (parent_depth == depth) & valid_edge
            edge_weight = depth_mask.to(pos_flat.dtype).unsqueeze(-1)

            parent_pos = pos_flat[safe_parent]
            index = depth * self.branching_factor + slot_idx
            onehot = torch.zeros_like(parent_pos).scatter(
                1, index.unsqueeze(1), 1.0
            )
            child_pos = (parent_pos + onehot) * edge_weight

            updates = torch.zeros_like(pos_flat)
            updates = updates.index_add(0, safe_child, child_pos)
            pos_flat = pos_flat + updates

        return pos_flat.view(tree_count, n_nodes, self.hidden_features)
