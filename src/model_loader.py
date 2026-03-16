from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from src.models.embedded_rnn import EmbeddedRNN
from src.models.tcn_bilstm import BiLSTM


@dataclass
class LoadedModel:
    model: nn.Module
    input_dim: int
    output_dim: int


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _build_bilstm_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> LoadedModel:
    # LSTM bidirectional: weight_ih_l0 has shape (4*hidden, input_dim)
    hidden_size = int(state_dict["rnn.weight_ih_l0"].shape[0]) // 4
    input_dim = int(state_dict["rnn.weight_ih_l0"].shape[1])
    output_dim = int(state_dict["fc.weight"].shape[0])

    model = BiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_size,
        output_dim=output_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    return LoadedModel(model=model, input_dim=input_dim, output_dim=output_dim)


def _build_embedded_rnn_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> LoadedModel:
    # weight_hh_l0 shape is [num_gates * hidden_size, hidden_size] — shape[1] gives true hidden_size
    hidden_size = int(state_dict["rnn.weight_hh_l0"].shape[1])
    input_dim = int(state_dict["rnn.weight_ih_l0"].shape[1])
    output_dim = int(state_dict["fc.weight"].shape[0])

    model = EmbeddedRNN(
        input_dim=input_dim,
        hidden_dim=hidden_size,
        output_dim=output_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    return LoadedModel(model=model, input_dim=input_dim, output_dim=output_dim)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> LoadedModel:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(ckpt)

    # BiLSTM: bidirectional LSTM has reverse weights
    if "rnn.weight_ih_l0_reverse" in state_dict and "fc.weight" in state_dict:
        loaded = _build_bilstm_from_state_dict(state_dict)
    # EmbeddedRNN: simple RNN without reverse weights
    elif "rnn.weight_ih_l0" in state_dict and "fc.weight" in state_dict:
        loaded = _build_embedded_rnn_from_state_dict(state_dict)
    else:
        sample_keys = list(state_dict.keys())[:20]
        raise ValueError(
            "Unsupported checkpoint architecture. Example keys: "
            + ", ".join(sample_keys)
        )

    loaded.model = loaded.model.to(device)
    loaded.model.eval()
    return loaded
