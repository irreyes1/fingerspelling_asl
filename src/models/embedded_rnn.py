import torch
import torch.nn as nn

class EmbeddedRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = self.log_softmax(out)
        out = out.permute(1, 0, 2)  # (T,B,C) for CTC
        return out
