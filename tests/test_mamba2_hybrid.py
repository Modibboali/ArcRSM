from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from torch import nn

from arc_agi_trm import build_trm
from arc_agi_trm.operators import Mamba2AttentionBlock, MambaDependencyError
from dataset.build_arc_dataset import DataProcessConfig, convert_dataset
from models.losses import ACTLossHead
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

from test_trm_smoke import synthetic_batch, tiny_config


class FakeMamba2(nn.Module):
    """Trainable, shape-preserving CPU mixer; it is not an SSM implementation."""

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        headdim: int,
        expand: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.expand = expand
        inner = d_model * expand
        self.in_proj = nn.Linear(d_model, inner, dtype=dtype)
        self.out_proj = nn.Linear(inner, d_model, dtype=dtype)
        self.forward_calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.out_proj(torch.nn.functional.silu(self.in_proj(hidden_states)))


def _hybrid_config() -> dict[str, Any]:
    config = tiny_config()
    config.update(mamba_d_state=4, mamba_headdim=8, mamba_expand=2)
    return config


@pytest.mark.parametrize(
    ("dtype", "forward_dtype"),
    [(torch.float32, "float32"), (torch.bfloat16, "bfloat16")],
)
def test_hybrid_block_preserves_shape_dtype_device_and_backpropagates(
    dtype: torch.dtype, forward_dtype: str
) -> None:
    config = tiny_config()
    config["forward_dtype"] = forward_dtype
    attention_model = build_trm(config)
    block = Mamba2AttentionBlock(
        attention_model.config,
        mamba_cls=FakeMamba2,
        d_state=4,
        headdim=8,
        expand=2,
    )
    hidden_states = torch.randn(2, 11, 32, dtype=dtype, requires_grad=True)
    cos_sin = attention_model.inner.rotary_emb()

    output = block(cos_sin=cos_sin, hidden_states=hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in block.parameters())
    assert (block.mamba1.d_state, block.mamba1.headdim, block.mamba1.expand) == (4, 8, 2)


def test_full_hybrid_trm_keeps_recursion_heads_loss_and_gradients() -> None:
    model = build_trm(_hybrid_config(), operator="mamba2_attention", mamba_cls=FakeMamba2)
    loss_model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    batch = synthetic_batch()
    carry = loss_model.initial_carry(batch)
    grad_modes: list[bool] = []

    def record_grad_mode(_module, _args):
        grad_modes.append(torch.is_grad_enabled())

    handle = model.inner.L_level.register_forward_pre_hook(record_grad_mode)
    try:
        new_carry, loss, metrics, outputs, _all_finished = loss_model(
            carry=carry,
            batch=batch,
            return_keys=("logits", "preds", "q_halt_logits"),
        )
    finally:
        handle.remove()

    block = model.inner.L_level.layers[0]
    assert grad_modes == ([False] * 6 + [True] * 3)
    assert block.mamba1.forward_calls == 9
    assert block.mamba2.forward_calls == 9
    assert new_carry.inner_carry.z_H.shape == (2, 11, 32)
    assert new_carry.inner_carry.z_L.shape == (2, 11, 32)
    assert not new_carry.inner_carry.z_H.requires_grad
    assert outputs is not None
    assert outputs["logits"].shape == (2, 9, 12)
    assert outputs["preds"].shape == (2, 9)
    assert outputs["q_halt_logits"].shape == (2,)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["lm_loss"])

    loss.backward()
    for mamba in (block.mamba1, block.mamba2):
        assert all(parameter.grad is not None for parameter in mamba.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in mamba.parameters())
    assert model.inner.lm_head.weight.grad is not None
    assert model.inner.q_head.weight.grad is not None


def test_missing_mamba_dependency_has_actionable_error(monkeypatch) -> None:
    import arc_agi_trm.operators as operators

    def missing_import(_name: str):
        raise ModuleNotFoundError("No module named 'mamba_ssm'")

    monkeypatch.setattr(operators.importlib, "import_module", missing_import)
    with pytest.raises(MambaDependencyError, match="docs/KAGGLE_MAMBA_INSTALL.md"):
        Mamba2AttentionBlock(
            build_trm(tiny_config()).config,
            d_state=4,
            headdim=8,
        )


def test_real_arc_encoded_batch_runs_through_hybrid_test_operator(tmp_path) -> None:
    source = tmp_path / "arc-files"
    built = tmp_path / "arc-arrays"
    source.mkdir()
    train_task = {
        "train": [{"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]}],
        "test": [{"input": [[4, 5], [6, 7]], "output": [[5, 4], [7, 6]]}],
    }
    for split in ("training", "evaluation"):
        (source / f"arc-agi_{split}_challenges.json").write_text(
            json.dumps({f"{split}-task": train_task}), encoding="utf-8"
        )
        (source / f"arc-agi_{split}_solutions.json").write_text(
            json.dumps({f"{split}-task": [[[5, 4], [7, 6]]]}), encoding="utf-8"
        )

    convert_dataset(
        DataProcessConfig(
            input_file_prefix=str(source / "arc-agi"),
            output_dir=str(built),
            subsets=["training", "evaluation"],
            test_set_name="evaluation",
            num_aug=0,
            seed=0,
        )
    )
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[str(built)],
            global_batch_size=1,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    _set_name, batch, _global_size = next(iter(dataset))

    config = {
        "batch_size": 1,
        "seq_len": dataset.metadata.seq_len,
        "puzzle_emb_ndim": 16,
        "num_puzzle_identifiers": dataset.metadata.num_puzzle_identifiers,
        "vocab_size": dataset.metadata.vocab_size,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 0,
        "L_layers": 1,
        "hidden_size": 16,
        "expansion": 1.0,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 1,
        "no_ACT_continue": True,
        "mamba_d_state": 4,
        "mamba_headdim": 8,
        "mamba_expand": 1,
    }
    model = build_trm(config, operator="mamba2_attention", mamba_cls=FakeMamba2)
    loss_model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    carry = loss_model.initial_carry(batch)
    carry, loss, _metrics, outputs, _all_finished = loss_model(
        carry=carry,
        batch=batch,
        return_keys=("logits", "preds", "q_halt_logits"),
    )

    assert batch["inputs"].shape == (1, 900)
    assert outputs is not None
    assert outputs["logits"].shape == (1, 900, 12)
    assert outputs["preds"].shape == (1, 900)
    assert carry.inner_carry.z_H.shape == (1, 901, 16)
    assert torch.isfinite(loss)
    loss.backward()
    block = model.inner.L_level.layers[0]
    assert all(p.grad is not None for p in block.mamba1.parameters())
    assert all(p.grad is not None for p in block.mamba2.parameters())
