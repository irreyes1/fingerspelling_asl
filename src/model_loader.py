import re
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.embedded_rnn import EmbeddedRNN
from src.models.tcn_bilstm import TCNBiRNN


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


def _infer_rnn_type_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    w_ih = state_dict.get("rnn.weight_ih_l0")
    w_hh = state_dict.get("rnn.weight_hh_l0")
    if w_ih is None or w_hh is None:
        raise KeyError("Missing rnn.weight_ih_l0 / rnn.weight_hh_l0 in checkpoint.")

    hidden = int(w_hh.shape[1])
    gates = int(w_ih.shape[0]) // hidden
    if gates == 4:
        return "lstm"
    if gates == 3:
        return "gru"
    return "rnn"


def _build_tcn_birnn_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> LoadedModel:
    input_proj_w = state_dict["input_proj.weight"]
    # Conv1d( in=input_dim, out=proj_dim, kernel=1 ) -> (out, in, 1)
    input_dim = int(input_proj_w.shape[1])
    proj_dim = int(input_proj_w.shape[0])

    tcn_idxs = set()
    for k in state_dict.keys():
        m = re.match(r"^tcn\.(\d+)\.net\.0\.weight$", k)
        if m:
            tcn_idxs.add(int(m.group(1)))
    if not tcn_idxs:
        raise KeyError("No TCN blocks found in checkpoint state_dict.")

    kernels = []
    for idx in sorted(tcn_idxs):
        w = state_dict[f"tcn.{idx}.net.0.weight"]
        kernels.append(int(w.shape[2]))

    hidden = int(state_dict["rnn.weight_hh_l0"].shape[1])
    output_dim = int(state_dict["classifier.weight"].shape[0])

    layer_ids = set()
    for k in state_dict.keys():
        m = re.match(r"^rnn\.weight_ih_l(\d+)$", k)
        if m:
            layer_ids.add(int(m.group(1)))
    rnn_layers = (max(layer_ids) + 1) if layer_ids else 1

    rnn_type = _infer_rnn_type_from_state_dict(state_dict)

    model = TCNBiRNN(
        input_dim=input_dim,
        proj_dim=proj_dim,
        tcn_kernels=tuple(kernels),
        rnn_hidden=hidden,
        rnn_layers=rnn_layers,
        rnn_type=rnn_type,
        output_dim=output_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    return LoadedModel(model=model, input_dim=input_dim, output_dim=output_dim)


def _build_embedded_rnn_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> LoadedModel:
    # weight_hh_l0 shape is [num_gates * hidden_size, hidden_size] — shape[1] gives true hidden_size
    hidden_size = int(state_dict["rnn.weight_hh_l0"].shape[1])
    input_dim = int(state_dict["rnn.weight_ih_l0"].shape[1])
    out_key = "fc.weight" if "fc.weight" in state_dict else "classifier.weight"
    output_dim = int(state_dict[out_key].shape[0])

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

    if "input_proj.weight" in state_dict and "classifier.weight" in state_dict:
        loaded = _build_tcn_birnn_from_state_dict(state_dict)
    elif "rnn.weight_ih_l0" in state_dict and ("fc.weight" in state_dict or "classifier.weight" in state_dict):
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
