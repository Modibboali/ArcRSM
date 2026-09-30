from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from arc_agi_trm import build_trm
from dataset.build_arc_dataset import np_grid_to_seq_translational_augment
from dataset.common import PuzzleDatasetMetadata
from evaluators.arc import ARC
from models.losses import ACTLossHead


def tiny_config() -> dict[str, object]:
    return {
        "batch_size": 2,
        "seq_len": 9,
        "puzzle_emb_ndim": 32,
        "num_puzzle_identifiers": 4,
        "vocab_size": 12,
        "H_cycles": 3,
        "L_cycles": 2,
        "H_layers": 0,
        "L_layers": 1,
        "hidden_size": 32,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 2,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 2,
        "no_ACT_continue": True,
    }


def synthetic_batch() -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.tensor(
            [[2, 3, 1, 4, 5, 1, 0, 0, 0], [3, 2, 1, 5, 4, 1, 0, 0, 0]],
            dtype=torch.int32,
        ),
        "labels": torch.tensor(
            [[3, 2, -100, 5, 4, -100, -100, -100, -100], [2, 3, -100, 4, 5, -100, -100, -100, -100]],
            dtype=torch.int32,
        ),
        "puzzle_identifiers": torch.tensor([1, 2], dtype=torch.int32),
    }


def test_model_construction_carry_forward_loss_and_backward() -> None:
    model = build_trm(tiny_config(), operator="attention")
    loss_model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    batch = synthetic_batch()

    carry = loss_model.initial_carry(batch)
    assert carry.inner_carry.z_H.shape == (2, 11, 32)
    assert carry.inner_carry.z_L.shape == (2, 11, 32)
    assert carry.halted.tolist() == [True, True]

    new_carry, loss, metrics, outputs, all_finished = loss_model(
        carry=carry,
        batch=batch,
        return_keys=("logits", "preds", "q_halt_logits"),
    )

    assert outputs is not None
    assert outputs["logits"].shape == (2, 9, 12)
    assert outputs["preds"].shape == (2, 9)
    assert outputs["q_halt_logits"].shape == (2,)
    assert new_carry.inner_carry.z_H.shape == (2, 11, 32)
    assert not new_carry.inner_carry.z_H.requires_grad
    assert torch.isfinite(loss)
    assert {"lm_loss", "q_halt_loss", "accuracy", "exact_accuracy"} <= metrics.keys()
    assert not bool(all_finished)

    loss.backward()
    block = model.inner.L_level.layers[0]
    assert block.self_attn.qkv_proj.weight.grad is not None
    assert model.inner.lm_head.weight.grad is not None
    assert model.inner.embed_tokens.embedding_weight.grad is not None


def test_only_final_full_recursion_tracks_gradients() -> None:
    model = build_trm(tiny_config())
    batch = synthetic_batch()
    carry = model.inner.reset_carry(
        torch.ones(2, dtype=torch.bool), model.inner.empty_carry(batch_size=2)
    )
    grad_modes: list[bool] = []

    def record_grad_mode(_module, _args):
        grad_modes.append(torch.is_grad_enabled())

    handle = model.inner.L_level.register_forward_pre_hook(record_grad_mode)
    try:
        new_carry, logits, q_logits = model.inner(carry, batch)
    finally:
        handle.remove()

    assert grad_modes == ([False] * 6 + [True] * 3)
    assert logits.shape == (2, 9, 12)
    assert q_logits[0].shape == (2,)
    assert q_logits[1].shape == (2,)
    assert not new_carry.z_H.requires_grad


def test_upstream_bfloat16_forward_on_cpu() -> None:
    config = tiny_config()
    config.update(H_cycles=1, L_cycles=1, forward_dtype="bfloat16")
    model = build_trm(config).eval()
    batch = synthetic_batch()

    carry, outputs = model(model.initial_carry(batch), batch)

    assert outputs["logits"].shape == (2, 9, 12)
    assert outputs["logits"].dtype == torch.bfloat16
    assert carry.inner_carry.z_H.dtype == torch.bfloat16


def test_factory_marks_experimental_operator_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="has not been implemented"):
        build_trm(tiny_config(), operator="experimental_2d_ssm")


def test_attention_factory_preserves_upstream_mlp_t_configuration() -> None:
    config = tiny_config()
    config["mlp_t"] = True

    model = build_trm(config, operator="attention")

    block = model.inner.L_level.layers[0]
    assert hasattr(block, "mlp_t")
    assert not hasattr(block, "self_attn")


def _encode(grid: list[list[int]]) -> torch.Tensor:
    arr = np.asarray(grid, dtype=np.uint8)
    encoded, _ = np_grid_to_seq_translational_augment(arr, arr, do_translation=False)
    return torch.from_numpy(encoded.astype(np.int32))


def test_arc_evaluator_accepts_predictions_and_votes(tmp_path) -> None:
    correct = [[1, 2], [3, 4]]
    wrong = [[4, 3], [2, 1]]
    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=12,
        seq_len=900,
        num_puzzle_identifiers=2,
        total_groups=1,
        mean_puzzle_examples=1.0,
        total_puzzles=1,
        sets=["all"],
    )
    (tmp_path / "identifiers.json").write_text('["<blank>", "task"]', encoding="utf-8")
    (tmp_path / "test_puzzles.json").write_text(
        '{"task": {"train": [], "test": [{"input": [[0]], "output": [[1, 2], [3, 4]]}]}}',
        encoding="utf-8",
    )

    evaluator = ARC(str(tmp_path), metadata, pass_Ks=(1, 2), submission_K=2)
    encoded_input = _encode([[0]]).repeat(3, 1)
    encoded_preds = torch.stack([_encode(wrong), _encode(wrong), _encode(correct)])
    batch = {
        "inputs": encoded_input,
        "puzzle_identifiers": torch.tensor([1, 1, 1], dtype=torch.int32),
    }
    preds = {
        "preds": encoded_preds,
        "q_halt_logits": torch.tensor([-2.0, -2.0, 10.0]),
    }
    evaluator.update_batch(batch, preds)

    def gather_object(value, output, **_kwargs):
        output[0] = value

    with patch("evaluators.arc.dist.gather_object", side_effect=gather_object):
        metrics = evaluator.result(str(tmp_path), rank=0, world_size=1)

    assert metrics == {"ARC/pass@1": 0.0, "ARC/pass@2": 1.0}
    assert (tmp_path / "submission.json").is_file()
