from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        return F.relu(h + self.skip(x))


class DAGAN(nn.Module):
    """Descriptor-Augmented Graph Attention Network."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        descriptor_dim: int,
        hidden_dim: int = 64,
        heads: int = 8,
        dropout: float = 0.2,
        descriptor_hidden_dim: int = 64,
        fusion_dim: int = 64,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        per_head = hidden_dim // heads
        self.gat1 = GATv2Conv(
            node_dim,
            per_head,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.gat2 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.desc_encoder = ResidualMLP(
            descriptor_dim, descriptor_hidden_dim, hidden_dim, dropout
        )

        # Atoms attend to the descriptor.
        self.atom_q = nn.Linear(hidden_dim, fusion_dim)
        self.desc_k = nn.Linear(hidden_dim, fusion_dim)
        self.desc_v = nn.Linear(hidden_dim, fusion_dim)
        self.atom_resid = nn.Linear(hidden_dim, fusion_dim)

        # The descriptor attends to atoms.
        self.back_q = nn.Linear(hidden_dim, fusion_dim)
        self.back_k = nn.Linear(hidden_dim, fusion_dim)
        self.back_v = nn.Linear(hidden_dim, fusion_dim)

        self.graph_proj = nn.Linear(fusion_dim, fusion_dim)
        self.desc_proj = nn.Linear(fusion_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = fusion_dim ** 0.5

    def encode_graph(self, x, edge_index, edge_attr):
        h = self.gat1(x, edge_index, edge_attr)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat2(h, edge_index, edge_attr)
        return F.elu(h)

    def forward(self, data, return_attention: bool = False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        desc = data.descriptors
        if desc.ndim == 1:
            desc = desc.view(-1, 1)
        # PyG batches descriptors as [B, D].
        desc_h = self.desc_encoder(desc)
        atom_h = self.encode_graph(x, edge_index, edge_attr)

        # Condition atom features on the descriptor.
        desc_for_atom = desc_h[batch]
        q_atom = self.atom_q(atom_h)
        k_desc = self.desc_k(desc_for_atom)
        v_desc = self.desc_v(desc_for_atom)
        forward_score = torch.sigmoid((q_atom * k_desc).sum(-1, keepdim=True) / self.scale)
        atom_att = self.atom_resid(atom_h) + forward_score * v_desc
        atom_att = F.relu(atom_att)

        # Pool atoms with descriptor-guided attention.
        q_back = self.back_q(desc_h)
        k_back = self.back_k(atom_h)
        v_back = self.back_v(atom_h)
        node_logits = (k_back * q_back[batch]).sum(-1) / self.scale

        # Normalize attention within each graph.
        weights = torch.zeros_like(node_logits)
        for graph_id in range(desc_h.size(0)):
            mask = batch == graph_id
            weights[mask] = torch.softmax(node_logits[mask], dim=0)

        graph_att = global_mean_pool(atom_att * weights.unsqueeze(-1), batch)
        # Convert the pooled mean to a weighted sum.
        counts = torch.bincount(batch, minlength=desc_h.size(0)).float().clamp_min(1).unsqueeze(-1)
        graph_att = graph_att * counts
        desc_att = torch.zeros_like(graph_att)
        for graph_id in range(desc_h.size(0)):
            mask = batch == graph_id
            desc_att[graph_id] = (weights[mask].unsqueeze(-1) * v_back[mask]).sum(dim=0)

        g = torch.tanh(self.graph_proj(graph_att))
        d = torch.tanh(self.desc_proj(desc_att))
        gate = torch.sigmoid(self.gate(torch.cat([g, d], dim=-1)))
        fused = gate * g + (1.0 - gate) * d
        logits = self.classifier(fused)

        if return_attention:
            return logits, {
                "forward_atom_gate": forward_score.squeeze(-1),
                "backward_atom_attention": weights,
                "fusion_gate": gate,
            }
        return logits


class DAGANNoCrossAttention(DAGAN):
    """DAGAN ablation without cross-attention."""

    def forward(self, data, return_attention: bool = False):
        desc_h = self.desc_encoder(data.descriptors)
        atom_h = self.encode_graph(data.x, data.edge_index, data.edge_attr)
        graph_h = global_mean_pool(atom_h, data.batch)
        g = torch.tanh(self.graph_proj(self.atom_resid(graph_h)))
        d = torch.tanh(self.desc_proj(self.desc_v(desc_h)))
        gate = torch.sigmoid(self.gate(torch.cat([g, d], dim=-1)))
        fused = gate * g + (1.0 - gate) * d
        logits = self.classifier(fused)
        return logits
