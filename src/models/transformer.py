import math

import torch
from torch import nn


class SalesPredictionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, num_heads=8, dropout=0.5):
        super().__init__()
        self.output_dim = output_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout), num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.fc(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.output_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SalesSequenceTransformer(nn.Module):
    """Transformer for sequence inputs shaped as (batch, seq_len, input_dim)."""

    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.5,
        output_dim=1,
        max_len=5000,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.dropout(x[:, -1, :])
        return self.fc_out(x)


class CentralTransformer(nn.Module):
    """Transformer with client embeddings for centralized multi-client training."""

    def __init__(
        self,
        input_dim,
        num_clients,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.5,
        output_dim=1,
        client_embed_dim=None,
        max_len=5000,
    ):
        super().__init__()
        client_embed_dim = client_embed_dim or d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.client_embed = nn.Embedding(num_clients, client_embed_dim)
        self.client_proj = nn.Identity() if client_embed_dim == d_model else nn.Linear(client_embed_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, inputs):
        x, client_ids = inputs
        x = self.input_proj(x)
        client_context = self.client_proj(self.client_embed(client_ids)).unsqueeze(1)
        x = self.pos_encoder(x + client_context)
        x = self.encoder(x)
        x = self.dropout(x[:, -1, :])
        return self.fc_out(x)
