import re
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn

from src.models.embedded_rnn import EmbeddedRNN
from src.models.tcn_bilstm import TCNBiRNN


@dataclass
class LoadedModel:
    model: nn.Module
    input_dim: int
    output_dim: int
    config: Dict[str, Any]


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


def _infer_subsampling_from_ckpt(
    state_dict: Dict[str, torch.Tensor],
    ckpt_config: Dict[str, Any],
) -> Dict[str, Any]:
    has_subsampling_keys = any(k.startswith("temporal_subsampling.layers.") for k in state_dict.keys())
    enabled = bool(ckpt_config.get("enable_temporal_subsampling", has_subsampling_keys))
    subsampling_type = str(ckpt_config.get("temporal_subsampling_type", "conv")).lower()
    factor = int(ckpt_config.get("temporal_subsampling_factor", 2))
    hidden_dim = ckpt_config.get("temporal_subsampling_hidden_dim", None)
    if hidden_dim is not None:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            hidden_dim = None

    if has_subsampling_keys and "temporal_subsampling_type" not in ckpt_config:
        # Conv layers have weights; avg/max pool do not.
        if any(re.match(r"^temporal_subsampling\.layers\.\d+\.weight$", k) for k in state_dict.keys()):
            subsampling_type = "conv"

    if has_subsampling_keys and "temporal_subsampling_factor" not in ckpt_config:
        conv_stage_ids = set()
        for k in state_dict.keys():
            m = re.match(r"^temporal_subsampling\.layers\.(\d+)\.weight$", k)
            if m:
                conv_stage_ids.add(int(m.group(1)))
        if conv_stage_ids:
            factor = 2 ** len(conv_stage_ids)

    return {
        "enable_temporal_subsampling": enabled,
        "temporal_subsampling_type": subsampling_type,
        "temporal_subsampling_factor": factor,
        "temporal_subsampling_hidden_dim": hidden_dim,
    }


def _build_tcn_birnn_from_state_dict(state_dict: Dict[str, torch.Tensor], ckpt_config: Dict[str, Any]) -> LoadedModel:
    input_proj_w = state_dict["input_proj.weight"]
    # Conv1d( in=post_subsampling_dim, out=proj_dim, kernel=1 ) -> (out, in, 1)
    post_subsample_dim = int(input_proj_w.shape[1])
    proj_dim = int(input_proj_w.shape[0])
    subsampling_cfg = _infer_subsampling_from_ckpt(state_dict, ckpt_config)

    input_dim = post_subsample_dim
    first_subsample_w = state_dict.get("temporal_subsampling.layers.0.weight")
    if (
        subsampling_cfg["enable_temporal_subsampling"]
        and subsampling_cfg["temporal_subsampling_type"] == "conv"
        and first_subsample_w is not None
    ):
        input_dim = int(first_subsample_w.shape[1])
        if subsampling_cfg["temporal_subsampling_hidden_dim"] is None:
            subsampling_cfg["temporal_subsampling_hidden_dim"] = int(first_subsample_w.shape[0])

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
    bidirectional = any(k.startswith("rnn.weight_ih_l0_reverse") for k in state_dict.keys())

    model = TCNBiRNN(
        input_dim=input_dim,
        proj_dim=proj_dim,
        tcn_kernels=tuple(kernels),
        rnn_hidden=hidden,
        rnn_layers=rnn_layers,
        rnn_type=rnn_type,
        output_dim=output_dim,
        bidirectional=bidirectional,
        enable_temporal_subsampling=subsampling_cfg["enable_temporal_subsampling"],
        temporal_subsampling_type=subsampling_cfg["temporal_subsampling_type"],
        temporal_subsampling_factor=subsampling_cfg["temporal_subsampling_factor"],
        temporal_subsampling_hidden_dim=subsampling_cfg["temporal_subsampling_hidden_dim"],
    )
    model.load_state_dict(state_dict, strict=True)
    return LoadedModel(model=model, input_dim=input_dim, output_dim=output_dim, config={})


def _build_embedded_rnn_from_state_dict(state_dict: Dict[str, torch.Tensor], ckpt_config: Dict[str, Any]) -> LoadedModel:
    rnn_type = _infer_rnn_type_from_state_dict(state_dict)
    hidden_size = int(state_dict["rnn.weight_hh_l0"].shape[1])
    post_subsample_dim = int(state_dict["rnn.weight_ih_l0"].shape[1])
    subsampling_cfg = _infer_subsampling_from_ckpt(state_dict, ckpt_config)
    input_dim = post_subsample_dim
    first_subsample_w = state_dict.get("temporal_subsampling.layers.0.weight")
    if (
        subsampling_cfg["enable_temporal_subsampling"]
        and subsampling_cfg["temporal_subsampling_type"] == "conv"
        and first_subsample_w is not None
    ):
        input_dim = int(first_subsample_w.shape[1])
        if subsampling_cfg["temporal_subsampling_hidden_dim"] is None:
            subsampling_cfg["temporal_subsampling_hidden_dim"] = int(first_subsample_w.shape[0])
    out_key = "fc.weight" if "fc.weight" in state_dict else "classifier.weight"
    output_dim = int(state_dict[out_key].shape[0])
    layer_ids = set()
    for k in state_dict.keys():
        m = re.match(r"^rnn\.weight_ih_l(\d+)$", k)
        if m:
            layer_ids.add(int(m.group(1)))
    rnn_layers = int(ckpt_config.get("num_layers", ckpt_config.get("rnn_layers", (max(layer_ids) + 1) if layer_ids else 1)))
    bidirectional = bool(
        ckpt_config.get("bidirectional", "bidir" in str(ckpt_config.get("wandb_tags", "")).lower())
        or any(k.startswith("rnn.weight_ih_l0_reverse") for k in state_dict.keys())
    )

    model = EmbeddedRNN(
        input_dim=input_dim,
        hidden_dim=hidden_size,
        output_dim=output_dim,
        rnn_type=rnn_type,
        num_layers=rnn_layers,
        bidirectional=bidirectional,
        enable_temporal_subsampling=subsampling_cfg["enable_temporal_subsampling"],
        temporal_subsampling_type=subsampling_cfg["temporal_subsampling_type"],
        temporal_subsampling_factor=subsampling_cfg["temporal_subsampling_factor"],
        temporal_subsampling_hidden_dim=subsampling_cfg["temporal_subsampling_hidden_dim"],
    )
    model.load_state_dict(state_dict, strict=True)
    return LoadedModel(model=model, input_dim=input_dim, output_dim=output_dim, config={})


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> LoadedModel:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(ckpt)
    ckpt_config: Dict[str, Any] = {}
    if isinstance(ckpt, dict):
        raw_cfg = ckpt.get("config", {})
        if isinstance(raw_cfg, dict):
            ckpt_config = raw_cfg

    if "input_proj.weight" in state_dict and "classifier.weight" in state_dict:
        loaded = _build_tcn_birnn_from_state_dict(state_dict, ckpt_config=ckpt_config)
    elif "rnn.weight_ih_l0" in state_dict and ("fc.weight" in state_dict or "classifier.weight" in state_dict):
        loaded = _build_embedded_rnn_from_state_dict(state_dict, ckpt_config=ckpt_config)
    else:
        sample_keys = list(state_dict.keys())[:20]
        raise ValueError(
            "Unsupported checkpoint architecture. Example keys: "
            + ", ".join(sample_keys)
        )

    loaded.config = ckpt_config
    loaded.model = loaded.model.to(device)
    loaded.model.eval()
    return loaded
