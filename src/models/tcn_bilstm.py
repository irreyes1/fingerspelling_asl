import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim, hidden_dim,
            batch_first=True, num_layers=2,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, input_lengths=None):
        T = x.size(1)
        if input_lengths is not None:
            # Pack: critical for bidirectional LSTM so backward pass
            # doesn't process padding before real data.
            packed = pack_padded_sequence(
                x, input_lengths.cpu().clamp(min=1, max=T),
                batch_first=True, enforce_sorted=False,
            )
            out, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            out, _ = self.rnn(x)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.log_softmax(out)
        out = out.permute(1, 0, 2)  # (T, B, C) for CTC
        return out
