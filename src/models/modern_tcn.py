"""
ModernTCN Architecture for Inertial Time-Series Navigation.

Implements modern temporal convolutional design:
- Large-kernel depthwise separable convolutions for extensive receptive field
- Inverted bottleneck MLPs with expansion ratio
- Layer normalization and residual skip connections
- High temporal efficiency and competitive representation learning
"""

import torch
import torch.nn as nn


class ModernTCNBlock(nn.Module):
    """Modern Temporal Convolutional Block with Inverted Bottleneck."""
    def __init__(self, channels: int, kernel_size: int = 17, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        # 1. Large-Kernel Depthwise 1D Convolution (Groups = Channels)
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels)  # Equivalent to LayerNorm across channel dimension
        
        # 2. Inverted Bottleneck MLP (1x1 Conv / Pointwise)
        hidden_dim = channels * expansion
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, channels, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L)"""
        residual = x
        out = self.dw_conv(x)
        out = self.norm(out)
        out = self.mlp(out)
        out = self.dropout(out)
        return residual + out


class ModernTCNModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        model_dim: int = 128,
        num_blocks: int = 3,
        kernel_size: int = 17,
        expansion: int = 4,
        out_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_dim = model_dim
        
        # Input stem: project 6 sensor channels to model_dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, model_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Stack of ModernTCN blocks
        self.blocks = nn.ModuleList([
            ModernTCNBlock(
                channels=model_dim,
                kernel_size=kernel_size,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        
        self.norm_out = nn.GroupNorm(1, model_dim)
        
        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, in_channels)
        Returns:
            Tensor of shape (Batch, out_dim)
        """
        # (B, L, C) -> (B, C, L)
        out = x.transpose(1, 2)
        out = self.stem(out)
        
        for block in self.blocks:
            out = block(out)
            
        out = self.norm_out(out)
        # Select last time step representation for velocity regression at endpoint
        last_step = out[:, :, -1]  # (B, model_dim)
        pred = self.head(last_step)
        return pred
