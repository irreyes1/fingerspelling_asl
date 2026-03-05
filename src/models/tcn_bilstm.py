from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, proj_dim, kernel_size=1)
        self.tcn = nn.ModuleList(
            [
                TemporalBlock(
                    channels=proj_dim,
                    kernel_size=k,
                    dilation=(2 ** i),
                    dropout=0.1,
                )
                for i, k in enumerate(tcn_kernels)
            ]
        )
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[rnn_type]
        self.rnn = rnn_cls(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Linear(rnn_hidden * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.input_proj(x)  # (B, C, T)
        for block in self.tcn:
            x = block(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)  # (T, B, C) for CTC
