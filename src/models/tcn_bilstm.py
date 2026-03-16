from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.temporal_subsampling import TemporalSubsampling


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] != x.shape[-1]:
            m = min(y.shape[-1], x.shape[-1])
            y = y[..., :m]
            x = x[..., :m]
        return self.norm(x + y)


class TCNBiRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        tcn_kernels: Tuple[int, ...],
        rnn_hidden: int,
        rnn_layers: int,
        rnn_type: str,
        output_dim: int,
        bidirectional: bool = True,
        dropout: float = 0.0,
        enable_temporal_subsampling: bool = False,
        temporal_subsampling_type: str = "conv",
        temporal_subsampling_factor: int = 2,
        temporal_subsampling_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        tcn_dropout = float(max(0.0, dropout))
        rnn_dropout = tcn_dropout if int(rnn_layers) > 1 else 0.0
        self.temporal_subsampling = TemporalSubsampling(
            input_dim=input_dim,
            enabled=enable_temporal_subsampling,
            subsampling_type=temporal_subsampling_type,
            factor=temporal_subsampling_factor,
            hidden_dim=temporal_subsampling_hidden_dim,
        )

        self.input_proj = nn.Conv1d(self.temporal_subsampling.output_dim, proj_dim, kernel_size=1)
        self.tcn = nn.ModuleList(
            [
                TemporalBlock(
                    channels=proj_dim,
                    kernel_size=k,
                    dilation=(2 ** i),
                    dropout=tcn_dropout,
                )
                for i, k in enumerate(tcn_kernels)
            ]
        )
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[rnn_type]
        self.rnn = rnn_cls(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=rnn_dropout,
        )
        rnn_out = rnn_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Linear(rnn_out, output_dim)

    def transform_input_lengths(self, input_lens: torch.Tensor) -> torch.Tensor:
        return self.temporal_subsampling.transform_input_lengths(input_lens)

    def forward(self, x: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.temporal_subsampling(x)  # (B, T_sub, H_sub) or identity
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.input_proj(x)  # (B, C, T)
        for block in self.tcn:
            x = block(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)  # (T, B, C) for CTC
