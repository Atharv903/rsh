"""
CNN-GRU Baseline Architecture for Inertial Navigation.

Extracts local spatial-temporal features using 1D convolutions,
models sequential dependencies using a multi-layer GRU,
and regresses 2D velocity vectors [v_east, v_north].
"""

import torch
import torch.nn as nn


class CNNGRUModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        conv_dims: list = [64, 128],
        kernel_sizes: list = [5, 3],
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        out_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_dims = conv_dims
        self.gru_hidden_dim = gru_hidden_dim
        
        # 1. 1D Temporal Convolutional Front-End
        conv_layers = []
        c_in = in_channels
        for c_out, k in zip(conv_dims, kernel_sizes):
            padding = k // 2
            conv_layers.extend([
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=padding),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            c_in = c_out
        self.conv_net = nn.Sequential(*conv_layers)
        
        # 2. Recurrent Sequence Encoder (GRU)
        self.gru = nn.GRU(
            input_size=c_in,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        
        # 3. Regression Head
        self.head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, in_channels)
        Returns:
            Tensor of shape (Batch, out_dim) representing 2D velocity [v_east, v_north]
        """
        # (B, L, C) -> (B, C, L) for 1D conv
        x = x.transpose(1, 2)
        feat = self.conv_net(x)
        # (B, C_out, L) -> (B, L, C_out) for GRU
        feat = feat.transpose(1, 2)
        
        out, _ = self.gru(feat)
        # Take the last time-step representation for velocity prediction at window endpoint
        last_step = out[:, -1, :]
        pred = self.head(last_step)
        return pred
