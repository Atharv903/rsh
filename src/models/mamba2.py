"""
Mamba-2 Architecture for Inertial Sequence Modeling.

Implements the official Mamba-2 / State Space Duality (SSD) principles (Dao & Gu, 2024):
- Selective Input-Dependent State Transitions (A, B, C, Delta)
- Continuous-time discretization via softplus step-size Delta
- Local 1D depthwise token mixing
- Multi-head selective recurrence dual to causal semi-separable matrix transformation (SSD)
- Pure PyTorch vectorized implementation (runs with high speed on both CPU and GPU)
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2Block(nn.Module):
    """
    Mamba-2 Selective State Space Duality (SSD) Block.
    
    Transforms input sequence via:
    1. LayerNorm
    2. Input projection into (x, z, B, C, dt)
    3. 1D depthwise convolution on x
    4. Multi-head State Space Duality (SSD) transformation
    5. Multiplicative SwiGLU gating
    6. Output projection with residual connection
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.n_heads = n_heads
        assert self.d_inner % n_heads == 0, f"d_inner ({self.d_inner}) must be divisible by n_heads ({n_heads})"
        self.head_dim = self.d_inner // n_heads
        
        # 1. Projections: u -> x (d_inner), z (d_inner), B (n_heads), C (n_heads), dt (n_heads)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + self.n_heads * 3)
        
        # 2. Local 1D Depthwise Convolution for short-term token mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # 3. Continuous-time SSM parameter: log(A), one per head
        # In Mamba-2 / SSD, A is scalar-times-identity per head (strictly negative real)
        a_init = torch.log(torch.linspace(0.1, 5.0, steps=self.n_heads))
        self.a_log = nn.Parameter(a_init)
        
        # Skip connection parameter D (one per channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 4. Out Projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input tensor of shape (Batch, Seq_Len, d_model)
        Returns:
            Output tensor of shape (Batch, Seq_Len, d_model)
        """
        B, L, _ = u.shape
        residual = u
        norm_u = self.norm(u)
        
        # Project: (B, L, d_inner*2 + n_heads*3)
        proj = self.in_proj(norm_u)
        
        # Split projections
        x_raw, z, B_mat, C_mat, dt = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.n_heads, self.n_heads, self.n_heads],
            dim=-1,
        )
        
        # Local 1D depthwise conv on x_raw: (B, L, d_inner) -> (B, d_inner, L)
        x_conv = self.conv1d(x_raw.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = F.silu(x_conv)  # (B, L, d_inner)
        
        # Discretize continuous step-size Delta: dt = softplus(dt)
        dt = F.softplus(dt)  # (B, L, n_heads)
        
        # SSM decay parameter A: A is strictly negative
        A = -torch.exp(self.a_log)  # (n_heads,)
        
        # Continuous-time decay across time steps:
        # dA[b, l, h] = dt[b, l, h] * A[h] <= 0
        dA = dt * A.view(1, 1, self.n_heads)  # (B, L, n_heads)
        
        # Cumulative decay: Lambda[b, l, h] = cumsum(dA)
        Lambda = torch.cumsum(dA, dim=1)  # (B, L, n_heads)
        
        # Pairwise decay matrix M[b, h, i, j] = exp(Lambda[i] - Lambda[j]) for j <= i
        # Transpose to (B, n_heads, L):
        Lambda_h = Lambda.permute(0, 2, 1)  # (B, n_heads, L)
        
        # (B, n_heads, L, 1) - (B, n_heads, 1, L) -> (B, n_heads, L, L)
        # Note: For j <= i, Lambda[i] <= Lambda[j] since A is negative, so diff <= 0.
        # For j > i, diff > 0 which could cause exp() to overflow to inf before masking!
        # Mask out j > i with -1e9 before exp to strictly prevent numerical overflow.
        diff = Lambda_h.unsqueeze(-1) - Lambda_h.unsqueeze(-2)
        diff = torch.clamp(diff, max=0.0)  # Bound above by 0
        causal_mask = torch.tril(torch.ones(L, L, device=u.device, dtype=torch.bool))
        diff = diff.masked_fill(~causal_mask, -1e9)
        decay = torch.exp(diff)
        decay = decay.masked_fill(~causal_mask, 0.0)
        
        # B_mat and C_mat modulation:
        # Kernel K[b, h, i, j] = C[b, i, h] * decay[b, h, i, j] * B[b, j, h]
        B_h = B_mat.permute(0, 2, 1)  # (B, n_heads, L)
        C_h = C_mat.permute(0, 2, 1)  # (B, n_heads, L)
        
        # (B, n_heads, L, 1) * (B, n_heads, L, L) * (B, n_heads, 1, L)
        K = C_h.unsqueeze(-1) * decay * B_h.unsqueeze(-2)  # (B, n_heads, L, L)
        
        # Reshape x to (B, n_heads, L, head_dim)
        x_reshaped = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply State Space Duality matrix multiplication:
        # y[b, h, i, d] = sum_j (K[b, h, i, j] * x[b, h, j, d])
        y_ssd = torch.matmul(K, x_reshaped)  # (B, n_heads, L, head_dim)
        
        # Flatten back to (B, L, d_inner)
        y_ssd = y_ssd.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_inner)
        
        # Add D skip connection: y = y + x * D
        y = y_ssd + x * self.D.unsqueeze(0).unsqueeze(0)
        
        # SwiGLU Gating: y = y * silu(z)
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        out = self.dropout(out)
        return residual + out


class Mamba2Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        d_model: int = 128,
        num_layers: int = 3,
        n_heads: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        out_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        
        # Input stem: project 6 IMU channels to d_model
        self.in_proj = nn.Linear(in_channels, d_model)
        
        # Mamba-2 SSD Layers
        self.layers = nn.ModuleList([
            Mamba2Block(
                d_model=d_model,
                n_heads=n_heads,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.norm_out = nn.LayerNorm(d_model)
        
        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
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
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h)
        
        # Endpoint velocity prediction
        last_step = h[:, -1, :]
        pred = self.head(last_step)
        return pred
