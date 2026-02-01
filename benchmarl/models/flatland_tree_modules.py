from __future__ import annotations

import torch
from torch import nn


class TreeLSTM(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_iou = nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = nn.Linear(3 * self.out_features, 3 * self.out_features, bias=False)
        self.W_c = nn.Linear(3 * self.out_features, self.out_features)

        self.W_f = nn.Linear(self.in_features, self.out_features)
        self.U_f = nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, forest, adjacency, node_order, edge_order):
        forest = forest.flatten(0, 2)
        adjacency_list = adjacency.flatten(0, 2)
        node_order = node_order.flatten(0, 2)
        edge_order = edge_order.flatten(0, 2)

        batch_size = node_order.shape[0]
        device = next(self.parameters()).device

        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        for n in range(node_order.max().item() + 1):
            self._run_lstm(n, h, c, forest, node_order, adjacency_list, edge_order)
        return h

    def _run_lstm(
        self,
        iteration: int,
        h: torch.Tensor,
        c: torch.Tensor,
        features: torch.Tensor,
        node_order: torch.Tensor,
        adjacency_list: torch.Tensor,
        edge_order: torch.Tensor,
    ):
        node_mask = node_order == iteration
        edge_mask = edge_order == iteration

        x = features[node_mask, :]

        if iteration == 0:
            iou = self.W_iou(x)
        else:
            mask = edge_mask.unsqueeze(-1)
            adjacency_masked = adjacency_list * mask
            parent_indexes = adjacency_masked[:, 0]
            child_indexes = adjacency_masked[:, 1]

            valid_mask = (
                (parent_indexes >= 0)
                & (parent_indexes < h.shape[0])
                & (child_indexes >= 0)
                & (child_indexes < h.shape[0])
            )
            safe_child = child_indexes.clamp(0, h.shape[0] - 1)

            child_h = h[safe_child, :]
            child_c = c[safe_child, :]
            if valid_mask.numel() > 0:
                child_h = child_h * valid_mask.unsqueeze(-1)
                child_c = child_c * valid_mask.unsqueeze(-1)

            i_dims = child_h.shape[0] // 3
            child_h_merge = child_h.unflatten(0, (i_dims, 3)).flatten(start_dim=1)
            iou = self.W_iou(x) + self.U_iou(child_h_merge)

        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            safe_parent = parent_indexes.clamp(0, features.shape[0] - 1)
            f = self.W_f(features[safe_parent, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            fc = f * child_c
            fc = fc.unflatten(0, (fc.shape[0] // 3, 3)).flatten(start_dim=1)
            c_reduce = self.W_c(fc)
            c[node_mask, :] = i * u + c_reduce

        h[node_mask, :] = o * torch.tanh(c[node_mask])


class TreeTransformer(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, n_nodes: int
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_nodes = n_nodes

        self.input_linear = nn.Linear(self.in_features, self.hidden_features)
        self.output_linear = nn.Linear(
            self.hidden_features * self.n_nodes, self.out_features
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_features,
            nhead=4,
            batch_first=True,
            dim_feedforward=self.hidden_features,
            dropout=0,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(self.out_features)

    def forward(self, forest: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, n_agents, n_nodes, _ = forest.shape
        embedded_forest = self.input_linear(forest)
        positional_encoding = self.get_positional_encoding(
            forest, adjacency, self.hidden_features
        )
        input_data = embedded_forest + positional_encoding
        input_data = input_data.reshape(
            (batch_size * n_agents, n_nodes, self.hidden_features)
        )
        idx = torch.randperm(n_nodes, device=input_data.device)
        input_data = input_data[:, idx]
        output = self.transformer_encoder(input_data)
        output = output.reshape(
            (batch_size * n_agents, n_nodes * self.hidden_features)
        )
        output = self.output_linear(output).reshape((batch_size, n_agents, -1))
        output = self.norm(output)
        return output

    def get_positional_encoding(self, forest, adjacency: torch.Tensor, output_dim):
        batch_size, n_agents, n_nodes, _ = forest.shape
        current_device = forest.device

        positional_encoding = torch.zeros(
            batch_size, n_agents, n_nodes, output_dim, device=current_device
        ).view(-1, output_dim)
        current_nodes = adjacency[:, :, 0, 0].flatten()
        adjacency_flat = (
            torch.cat(
                (
                    -1 * torch.ones(batch_size, n_agents, 1, 3, device=current_device),
                    adjacency,
                ),
                2,
            )
            .view(-1, 3)
            .type(torch.int64)
        )
        current_nodes = adjacency_flat[torch.isin(adjacency_flat[:, 0], current_nodes)][
            :, 1
        ]
        parent_nodes = torch.tensor([], device=current_device)
        current_depth = 0

        while True:
            parent_nodes = adjacency_flat[current_nodes, 0]
            index_tensor = torch.tensor(
                [current_depth * 3, current_depth * 3 + 1, current_depth * 3 + 2],
                device=current_device,
            )
            index_tensor = index_tensor.repeat(len(current_nodes) // 3)
            positional_encoding[current_nodes] = positional_encoding[parent_nodes]
            positional_encoding[current_nodes, index_tensor] = 1

            current_nodes = adjacency[
                torch.isin(adjacency[:, :, :, 0], current_nodes)
            ][:, 1]
            current_depth += 1

            if torch.numel(current_nodes) == 0:
                break

        positional_encoding = positional_encoding.view(
            batch_size, n_agents, n_nodes, output_dim
        )
        return positional_encoding
