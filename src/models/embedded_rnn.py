from typing import Optional

import torch
import torch.nn as nn

from src.models.temporal_subsampling import TemporalSubsampling

class EmbeddedRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        rnn_type: str = "rnn",
        num_layers: int = 1,
        bidirectional: bool = False,
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
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[str(rnn_type).lower()]
        rnn_dropout = 0.0 if int(num_layers) <= 1 else 0.0
        self.rnn = rnn_cls(
            self.temporal_subsampling.output_dim,
            hidden_dim,
            num_layers=int(num_layers),
            bidirectional=bool(bidirectional),
            batch_first=True,
            dropout=rnn_dropout,
        )
        rnn_out = int(hidden_dim) * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_out, output_dim)
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
