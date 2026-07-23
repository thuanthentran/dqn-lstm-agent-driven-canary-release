"""TransformerFeatureExtractor for Stable-Baselines3 PPO.

Provides explainable AI through two attention mechanisms:
  - Feature Attention: which of the 5 input features matter most at each timestep
  - Temporal Attention: which past timesteps the agent is attending to

Input observation shape: (batch, seq_len=30, n_features=5)
Output: (batch, features_dim=d_model)
"""

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureAttentionLayer(nn.Module):
    """Per-timestep cross-attention over input features.

    For each timestep, treats the 5 raw feature values as 5 tokens and
    uses a learnable aggregation query to attend over them.

    Input:  (B, T, n_features)
    Output: (B, T, d_model), feature_weights (B, n_heads, T, n_features)
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int = 1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads

        # Project each scalar feature value → d_model vector
        self.feature_proj = nn.Linear(1, d_model)
        # Learnable feature-type embeddings (one per feature channel)
        self.feature_type_embed = nn.Parameter(torch.randn(1, 1, n_features, d_model) * 0.02)
        # Learnable aggregation query
        self.agg_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Cross-attention: query attends over feature tokens
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C) raw feature values
        Returns:
            out:     (B, T, d_model)
            weights: (B, n_heads, T, C) attention weights per feature
        """
        B, T, C = x.shape

        # Project each feature scalar to d_model: (B, T, C, 1) → (B, T, C, d_model)
        feat_tokens = self.feature_proj(x.unsqueeze(-1))
        # Add learnable feature-type embeddings
        feat_tokens = feat_tokens + self.feature_type_embed

        # Reshape: merge B and T → (B*T, C, d_model)
        feat_tokens = feat_tokens.reshape(B * T, C, self.d_model)

        # Expand aggregation query: (B*T, 1, d_model)
        query = self.agg_query.expand(B * T, -1, -1)

        # Cross-attention: Q=(B*T,1,d), K=V=(B*T,C,d) → out=(B*T,1,d), w=(B*T,n_heads,1,C)
        out, weights = self.cross_attn(
            query, feat_tokens, feat_tokens, average_attn_weights=False
        )

        out = out.squeeze(1).reshape(B, T, self.d_model)  # (B, T, d_model)
        out = self.norm(out)

        # Reshape weights: (B*T, n_heads, 1, C) → (B, T, n_heads, C) → (B, n_heads, T, C)
        weights = weights.squeeze(2).reshape(B, T, self.n_heads, C)
        weights = weights.permute(0, 2, 1, 3)  # (B, n_heads, T, C)

        return out, weights


class TemporalTransformerEncoder(nn.Module):
    """Transformer Encoder with attention weight capture.

    Standard self-attention over the time dimension with hooks to
    capture attention weights for explainability.

    Input:  (B, T, d_model)
    Output: (B, T, d_model), temporal_weights (B, n_heads, T, T)
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TemporalTransformerBlock(d_model, n_heads, dropout)
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
            attn_weights: (B, n_heads, T, T) from the LAST layer
        """
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x)
        return x, attn_weights


class TemporalTransformerBlock(nn.Module):
    """Single Transformer block: self-attention + FFN with residual connections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        # Self-attention with residual
        attn_out, attn_weights = self.self_attn(
            x, x, x, average_attn_weights=False
        )
        x = self.norm1(x + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x, attn_weights  # attn_weights: (B, n_heads, T, T)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """SB3-compatible Transformer feature extractor with explainable attention.

    Architecture:
        1. Feature Attention: cross-attention aggregating 5 features per timestep
        2. Positional Encoding: learnable position embeddings
        3. Temporal Transformer: self-attention over the time axis
        4. Mean Pooling → features_dim output

    Explainability attributes (updated every forward pass):
        - last_feature_attention:  (B, n_heads_feat, T, n_features)
        - last_temporal_attention: (B, n_heads_temp, T, T)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        d_model: int = 64,
        n_heads: int = 4,
        n_heads_feature: int = 1,
        n_layers: int = 2,
        seq_len: int = 30,
        n_features: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim=d_model)

        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        # --- Feature Attention ---
        self.feature_attention = FeatureAttentionLayer(n_features, d_model, n_heads=n_heads_feature)

        # --- Positional Encoding (learnable) ---
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # --- Temporal Transformer Encoder ---
        self.temporal_encoder = TemporalTransformerEncoder(d_model, n_heads, n_layers, dropout)

        # --- XAI storage (detached, moved to CPU after forward) ---
        self.last_feature_attention: Optional[torch.Tensor] = None
        self.last_temporal_attention: Optional[torch.Tensor] = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (B, seq_len, n_features) = (B, 30, 5)
        Returns:
            features: (B, d_model) = (B, 64)
        """
        # 1. Feature Attention
        x, feat_attn = self.feature_attention(observations)  # x: (B,T,d), feat_attn: (B,nh,T,5)

        # 2. Add positional encoding
        x = x + self.pos_embed

        # 3. Temporal Transformer Encoder
        x, temp_attn = self.temporal_encoder(x)  # x: (B,T,d), temp_attn: (B,nh,T,T)

        # 4. Mean pooling over time
        features = x.mean(dim=1)  # (B, d_model)

        # 5. Store attention maps for XAI (detach to avoid graph retention)
        self.last_feature_attention = feat_attn.detach().cpu()
        self.last_temporal_attention = temp_attn.detach().cpu()

        return features

    def get_attention_maps(self) -> Dict[str, Optional[np.ndarray]]:
        """Return the latest attention weights as numpy arrays.

        Returns:
            dict with keys:
                - 'feature_attention':  np.ndarray (B, n_heads, T, n_features) or None
                - 'temporal_attention': np.ndarray (B, n_heads, T, T) or None
        """
        fa = self.last_feature_attention.numpy() if self.last_feature_attention is not None else None
        ta = self.last_temporal_attention.numpy() if self.last_temporal_attention is not None else None
        return {"feature_attention": fa, "temporal_attention": ta}