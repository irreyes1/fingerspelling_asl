from typing import Optional

import torch
import torch.nn as nn


class TemporalSubsampling(nn.Module):
    """
    Optional temporal compression block for inputs shaped (B, T, F).
    It applies stride-2 layers repeatedly until reaching the requested factor.
    """

    def __init__(
        self,
        input_dim: int,
        enabled: bool = False,
        subsampling_type: str = "conv",
        factor: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.subsampling_type = str(subsampling_type).lower()
        self.factor = int(factor)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else None

        if not self.enabled:
            self.output_dim = self.input_dim
            self.num_stages = 0
            self.layers = nn.ModuleList()
            return

        if self.subsampling_type not in {"conv", "avgpool", "maxpool"}:
            raise ValueError(f"Unsupported subsampling type: {self.subsampling_type}")
        if self.factor not in {2, 4}:
            raise ValueError(f"Unsupported subsampling factor: {self.factor}; expected one of [2, 4]")

        self.num_stages = 1 if self.factor == 2 else 2
        self.layers = nn.ModuleList()

        channels_in = self.input_dim
        channels_hidden = self.hidden_dim if self.hidden_dim is not None else self.input_dim

        for i in range(self.num_stages):
            if self.subsampling_type == "conv":
                channels_out = channels_hidden
                layer = nn.Conv1d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                self.layers.append(layer)
                if i < self.num_stages - 1:
                    self.layers.append(nn.ReLU())
                channels_in = channels_out
            elif self.subsampling_type == "avgpool":
                self.layers.append(nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
            else:
                self.layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

        self.output_dim = channels_in

    def transform_input_lengths(self, input_lens: torch.Tensor) -> torch.Tensor:
        out = input_lens.clone().long()
        if not self.enabled:
            return out
        for _ in range(self.num_stages):
            # kernel=3, stride=2, padding=1 -> floor((L + 1) / 2)
            out = ((out + 1) // 2).clamp_min(1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        # (B, T, F) -> (B, F, T) for Conv1d/Pool1d temporal ops.
        y = x.transpose(1, 2)
        for layer in self.layers:
            y = layer(y)
        # Back to (B, T_sub, H)
        return y.transpose(1, 2)
