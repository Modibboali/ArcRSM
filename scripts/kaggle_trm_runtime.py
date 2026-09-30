"""Shared setup helpers for the manually run Kaggle GPU scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "external" / "TinyRecursiveModels",
):
    text_path = str(import_path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

from arc_agi_trm import build_trm  # noqa: E402
from models.losses import ACTLossHead  # noqa: E402


def make_config(
    *,
    batch_size: int,
    hidden_size: int = 512,
    seq_len: int = 900,
    h_cycles: int = 2,
    l_cycles: int = 2,
    layers: int = 1,
    heads: int = 8,
    forward_dtype: str = "bfloat16",
    mamba_d_state: int = 128,
    mamba_headdim: int = 64,
    mamba_expand: int = 2,
) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "puzzle_emb_ndim": hidden_size,
        "num_puzzle_identifiers": 8,
        "vocab_size": 12,
        "H_cycles": h_cycles,
        "L_cycles": l_cycles,
        "H_layers": 0,
        "L_layers": layers,
        "hidden_size": hidden_size,
        "expansion": 4.0,
        "num_heads": heads,
        "pos_encodings": "rope",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "forward_dtype": forward_dtype,
        "mlp_t": False,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
        "mamba_d_state": mamba_d_state,
        "mamba_headdim": mamba_headdim,
        "mamba_expand": mamba_expand,
    }


def make_batch(*, batch_size: int, seq_len: int, device: torch.device):
    inputs = torch.randint(0, 12, (batch_size, seq_len), dtype=torch.int32, device=device)
    labels = torch.randint(0, 12, (batch_size, seq_len), dtype=torch.int32, device=device)
    if seq_len > 1:
        labels[:, -1] = -100
    puzzle_identifiers = torch.randint(
        1, 8, (batch_size,), dtype=torch.int32, device=device
    )
    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
    }


def build_loss_model(config: dict[str, Any], *, operator: str, device: torch.device):
    # Match upstream pretrain.py's device-context construction path.
    with torch.device(device):
        model = build_trm(config, operator=operator)
        loss_model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    return loss_model.to(device)


def model_step(loss_model, batch, *, return_outputs: bool = True):
    carry = loss_model.initial_carry(batch)
    return loss_model(
        carry=carry,
        batch=batch,
        return_keys=("logits", "preds", "q_halt_logits") if return_outputs else (),
    )


def check_training_result(result, *, batch_size: int, seq_len: int, vocab_size: int = 12):
    carry, loss, metrics, outputs, _all_finished = result
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("TRM loss is not finite")
    if outputs is None or outputs["logits"].shape != (batch_size, seq_len, vocab_size):
        raise RuntimeError("Unexpected lm_head output shape")
    if outputs["q_halt_logits"].shape != (batch_size,):
        raise RuntimeError("Unexpected q_head output shape")
    if carry.inner_carry.z_H.shape[0] != batch_size:
        raise RuntimeError("Unexpected z_H carry batch dimension")
    if not bool(torch.isfinite(outputs["logits"]).all()):
        raise RuntimeError("TRM logits contain non-finite values")
    return carry, loss, metrics, outputs


def assert_finite_gradients(module: torch.nn.Module) -> int:
    parameters = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    missing = [name for name, p in parameters if p.grad is None]
    if missing:
        raise RuntimeError(f"Missing gradients for trainable parameters: {missing}")
    nonfinite = [
        name
        for name, p in parameters
        if not bool(torch.isfinite(p.grad).all())
    ]
    if nonfinite:
        raise RuntimeError(f"Non-finite gradients for parameters: {nonfinite}")
    return len(parameters)


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def memory_report(device: torch.device) -> dict[str, int]:
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
