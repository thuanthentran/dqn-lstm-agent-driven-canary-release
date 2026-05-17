import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOTCNPolicy(nn.Module):
    """TCN feature extractor with actor-critic heads for PPO.

    Input: (batch, channels=5, seq_len)
    Outputs:
      - logits: (batch, action_dim)
      - value: (batch, 1)
    """

    def __init__(self, in_channels: int = 5, seq_len: int = 30, action_dim: int = 3, tcn_channels=(16, 32, 64, 128), kernel_size: int = 3, hidden_dim: int = 256):
        super(PPOTCNPolicy, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.action_dim = action_dim

        layers = []
        prev_ch = in_channels
        dilations = [1, 2, 4, 8][: len(tcn_channels)]
        for out_ch, dilation in zip(tcn_channels, dilations):
            pad = (kernel_size - 1) * dilation
            conv = nn.Conv1d(prev_ch, out_ch, kernel_size, dilation=dilation)
            # We'll apply left (causal) padding manually in forward
            layers.append((conv, dilation, pad))
            prev_ch = out_ch

        self.tcn_layers = nn.ModuleList([l[0] for l in layers])
        self.tcn_dilations = [l[1] for l in layers]
        self.tcn_pads = [l[2] for l in layers]

        # After TCN take last timestep representation and pass through MLPs
        final_ch = prev_ch
        self.shared_fc = nn.Linear(final_ch, hidden_dim)

        # Actor head
        self.actor_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic head
        self.critic_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _forward_tcn(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = x
        for conv, dilation, pad in zip(self.tcn_layers, self.tcn_dilations, self.tcn_pads):
            # causal pad on the left
            out = F.pad(out, (pad, 0))
            out = conv(out)
            out = F.relu(out)

        return out

    def forward(self, x: torch.Tensor):
        """Return logits and state-value.

        Args:
            x: (B, C, T)
        Returns:
            logits: (B, action_dim), value: (B, 1)
        """
        # ensure shape
        if x.ndim != 3:
            raise ValueError("Expected input (B, C, T)")

        tcn_out = self._forward_tcn(x)  # (B, final_ch, T)
        # use last time-step representation (causal)
        last = tcn_out[:, :, -1]  # (B, final_ch)

        shared = F.relu(self.shared_fc(last))

        logits = self.actor_fc(shared)
        value = self.critic_fc(shared)

        return logits, value.squeeze(-1)