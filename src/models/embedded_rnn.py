import torch
import torch.nn as nn
from typing import Optional

from src.models.temporal_subsampling import TemporalSubsampling

class EmbeddedRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        enable_temporal_subsampling: bool = False,
        temporal_subsampling_type: str = "conv",
        temporal_subsampling_factor: int = 2,
        temporal_subsampling_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.temporal_subsampling = TemporalSubsampling(
            input_dim=input_dim,
            enabled=enable_temporal_subsampling,
            subsampling_type=temporal_subsampling_type,
            factor=temporal_subsampling_factor,
            hidden_dim=temporal_subsampling_hidden_dim,
        )
        self.rnn = nn.RNN(self.temporal_subsampling.output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def transform_input_lengths(self, input_lens: torch.Tensor) -> torch.Tensor:
        return self.temporal_subsampling.transform_input_lengths(input_lens)

    def forward(self, x):
        x = self.temporal_subsampling(x)  # (B, T_sub, H_sub) or identity
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = self.log_softmax(out)
        out = out.permute(1, 0, 2)  # (T,B,C) for CTC
        return out
