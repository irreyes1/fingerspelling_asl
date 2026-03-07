import torch
import torch.nn as nn

from src.models.embedded_rnn import EmbeddedRNN
from src.models.tcn_bilstm import TCNBiRNN
from src.models.temporal_subsampling import TemporalSubsampling


def test_temporal_subsampling_conv_shape_and_lengths():
    block = TemporalSubsampling(
        input_dim=63,
        enabled=True,
        subsampling_type="conv",
        factor=2,
        hidden_dim=48,
    )
    x = torch.randn(3, 17, 63)
    y = block(x)
    assert y.shape == (3, 9, 48)

    in_lens = torch.tensor([17, 16, 1], dtype=torch.long)
    out_lens = block.transform_input_lengths(in_lens)
    assert out_lens.tolist() == [9, 8, 1]


def test_temporal_subsampling_pool_factor4_lengths():
    block = TemporalSubsampling(
        input_dim=63,
        enabled=True,
        subsampling_type="avgpool",
        factor=4,
    )
    in_lens = torch.tensor([17, 16, 7, 1], dtype=torch.long)
    out_lens = block.transform_input_lengths(in_lens)
    assert out_lens.tolist() == [5, 4, 2, 1]


def test_ctc_compatibility_with_subsampling():
    model = TCNBiRNN(
        input_dim=63,
        proj_dim=32,
        tcn_kernels=(3, 3),
        rnn_hidden=16,
        rnn_layers=1,
        rnn_type="gru",
        output_dim=30,
        bidirectional=True,
        enable_temporal_subsampling=True,
        temporal_subsampling_type="conv",
        temporal_subsampling_factor=2,
        temporal_subsampling_hidden_dim=32,
    )
    x = torch.randn(2, 20, 63)
    in_lens = torch.tensor([20, 13], dtype=torch.long)
    out_lens = model.transform_input_lengths(in_lens)

    log_probs = model(x)  # (T, B, C)
    assert log_probs.shape[0] == int(out_lens.max().item())

    # Build valid CTC targets where U <= transformed T for each sample.
    targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    target_lens = torch.tensor([3, 3], dtype=torch.long)
    loss = nn.CTCLoss(blank=0, zero_infinity=True)(log_probs, targets, out_lens, target_lens)
    assert torch.isfinite(loss).item()


def test_padding_case_with_embedded_rnn_subsampling():
    model = EmbeddedRNN(
        input_dim=16,
        hidden_dim=12,
        output_dim=8,
        enable_temporal_subsampling=True,
        temporal_subsampling_type="maxpool",
        temporal_subsampling_factor=2,
    )
    x = torch.zeros(2, 10, 16)
    x[0, :7] = torch.randn(7, 16)
    x[1, :3] = torch.randn(3, 16)
    in_lens = torch.tensor([7, 3], dtype=torch.long)
    out_lens = model.transform_input_lengths(in_lens)
    assert out_lens.tolist() == [4, 2]

    out = model(x)
    assert out.shape[0] >= int(out_lens.max().item())
