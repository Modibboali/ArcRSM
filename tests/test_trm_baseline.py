from __future__ import annotations

import builtins
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arc_agi_trm import build_trm, load_pretrained_model, save_pretrained
from arc_agi_trm.experiment import load_experiment_config, train_one_epoch
from arc_agi_trm.progress import TrainingProgress
from arc_agi_trm.runner import estimate_train_optimizer_updates
from models.losses import ACTLossHead
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from scripts.train_mamba2_hybrid import _load_resume, _save_checkpoint, _validate_config
from test_trm_smoke import synthetic_batch, tiny_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _ProgressBar:
    instances: list["_ProgressBar"] = []
    messages: list[str] = []

    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.n = kwargs["initial"]
        self.postfix = {}
        self.closed = False
        self.instances.append(self)

    def set_description(self, _description):
        pass

    def update(self, amount):
        self.n += amount

    def set_postfix(self, values, refresh=False):
        self.postfix = dict(values)

    def refresh(self):
        pass

    def close(self):
        self.closed = True

    @classmethod
    def write(cls, message):
        cls.messages.append(message)


def test_attention_operator_constructs_original_upstream_blocks_without_mamba_import(monkeypatch):
    original_import = builtins.__import__
    attempted_mamba_imports: list[str] = []

    def guarded_import(name, *args, **kwargs):
        if (
            name == "mamba_ssm"
            or name.startswith("mamba_ssm.")
            or name == "arc_agi_trm.operators"
        ):
            attempted_mamba_imports.append(name)
            raise AssertionError(f"original TRM attempted to import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    model = build_trm(tiny_config(), operator="attention")

    block = model.inner.L_level.layers[0]
    assert type(block).__name__ == "TinyRecursiveReasoningModel_ACTV1Block"
    assert hasattr(block, "self_attn")
    assert hasattr(block, "mlp")
    assert attempted_mamba_imports == []


def test_original_trm_optimizer_progress_and_pretrained_checkpoint_roundtrip(tmp_path: Path):
    model_config = tiny_config()
    trm = build_trm(model_config, operator="attention")
    loss_model = ACTLossHead(trm, loss_type="stablemax_cross_entropy")
    optimizer = torch.optim.AdamW(loss_model.parameters(), lr=1e-3)
    progress = TrainingProgress(
        total_steps=1,
        initial_step=0,
        initial_epoch=1,
        total_epochs=1,
        tqdm_factory=_ProgressBar,
    )
    updates: list[dict] = []

    def on_step(update):
        updates.append(update)
        progress.optimizer_step(update)

    result = train_one_epoch(
        model=loss_model,
        loader=[("all", synthetic_batch(), 2)],
        optimizers=[optimizer],
        optimizer_base_lrs=[1e-3],
        device=torch.device("cpu"),
        carry=None,
        optimizer_step=0,
        total_steps=1,
        warmup_steps=0,
        min_lr_ratio=1.0,
        global_batch_size=2,
        on_optimizer_step=on_step,
    )
    assert result.optimizer_step == 1
    assert len(updates) == 1
    assert math.isfinite(updates[0]["training_loss"])
    assert updates[0]["examples"] == 2
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in loss_model.parameters()
    )
    bar = _ProgressBar.instances[-1]
    assert bar.n == bar.total == 1
    assert {"loss", "lr", "ex/s", "eta~"} <= bar.postfix.keys()

    progress.epoch_completed(
        1,
        {"training_grid_accuracy": 0.5, "training_batch_exact_accuracy": 0.25},
    )
    assert set(bar.postfix) == {"loss", "lr", "ex/s", "eta~"}
    progress.close(final_step=1, status="complete")
    assert bar.closed
    assert set(bar.postfix) == {"loss", "lr", "ex/s", "eta~"}

    config = {
        "operator": "attention",
        "seed": 7,
        "epochs": 1,
        "data": {"num_aug": 0, "validation_tasks": 1},
        "model": {
            **model_config,
            "seq_len": 9,
            "vocab_size": 12,
            "num_puzzle_identifiers": 4,
        },
        "optimization": {
            "global_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "lr": 1e-3,
            "puzzle_emb_lr": 1e-3,
            "lr_min_ratio": 1.0,
            "lr_warmup_steps": 0,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.0,
            "puzzle_emb_weight_decay": 0.0,
            "gradient_clip_norm": None,
        },
        "run": {
            "eval_every_epochs": 1,
            "checkpoint_every_epochs": 1,
            "eval_attempts": 1,
            "num_workers": 0,
        },
        "estimated_total_optimizer_steps": 1,
    }

    class DatasetState:
        _iters = 1

    dataset = DatasetState()
    _save_checkpoint(
        run_dir=tmp_path,
        model=loss_model,
        optimizers=[optimizer],
        config=config,
        epoch=1,
        optimizer_step=result.optimizer_step,
        best_metrics={},
        train_dataset=dataset,
        carry=result.carry,
        rank=0,
        world_size=1,
        cpu_group=None,
        timing_summary={"epoch_duration_sum": 1.0, "epoch_duration_count": 1},
        is_best=False,
    )
    restored_trm = build_trm(model_config, operator="attention")
    restored_model = ACTLossHead(restored_trm, loss_type="stablemax_cross_entropy")
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_dataset = DatasetState()
    restored_dataset._iters = 0
    epoch, step, restored_carry, _best, _timing = _load_resume(
        path=tmp_path / "latest.pt",
        model=restored_model,
        optimizers=[restored_optimizer],
        config=config,
        train_dataset=restored_dataset,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    assert (epoch, step, restored_dataset._iters) == (1, 1, 1)
    assert restored_carry.current_data["inputs"].shape == synthetic_batch()["inputs"].shape

    artifact = save_pretrained(
        loss_model,
        tmp_path / "trm-baseline.pt",
        {"operator": "attention", "model": model_config},
    )
    restored = load_pretrained_model(artifact, "cpu")
    assert type(restored.model.inner.L_level.layers[0]) is type(
        loss_model.model.inner.L_level.layers[0]
    )
    assert restored.model._arc_agi_trm_metadata["operator"] == "attention"


def _fake_upstream_dataset() -> PuzzleDataset:
    dataset = PuzzleDataset.__new__(PuzzleDataset)
    dataset.config = PuzzleDatasetConfig(
        seed=17,
        dataset_paths=[],
        global_batch_size=4,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset.local_batch_size = 4
    dataset.metadata = SimpleNamespace(
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
    )
    puzzle_indices = np.asarray([0, 3, 6, 7, 11, 14, 16, 20, 23, 25], dtype=np.int64)
    dataset._data = {
        "all": {
            "inputs": np.zeros((25, 2), dtype=np.int32),
            "labels": np.zeros((25, 2), dtype=np.int32),
            "puzzle_identifiers": np.arange(9, dtype=np.int32),
            "puzzle_indices": puzzle_indices,
            "group_indices": np.asarray([0, 2, 4, 6, 7, 9], dtype=np.int64),
        }
    }
    dataset._iters = 0
    dataset._lazy_load_dataset = lambda: None
    return dataset


def test_optimizer_update_count_matches_actual_upstream_sampler_epochs():
    dataset = _fake_upstream_dataset()
    expected = estimate_train_optimizer_updates(
        dataset,
        epochs=4,
        accumulation_steps=2,
    )
    assert all(updates > 0 for updates in expected)
    assert sum(expected) > 0
    assert dataset._iters == 0

    actual = []
    for _ in range(4):
        batches = list(dataset._iter_train())
        actual.append((len(batches) + 1) // 2)
    assert expected == actual


def test_original_and_matched_baseline_presets_are_distinct():
    faithful = load_experiment_config(PROJECT_ROOT / "configs" / "trm_baseline_100_epochs.json")
    matched = load_experiment_config(
        PROJECT_ROOT / "configs" / "trm_matched_mamba2_mlp_100_epochs.json"
    )

    assert faithful["operator"] == matched["operator"] == "attention"
    assert faithful["expected_world_size"] == matched["expected_world_size"] == 4
    assert faithful["model"]["L_cycles"] == 6
    assert faithful["model"]["pos_encodings"] == "rope"
    assert faithful["optimization"]["global_batch_size"] == 768
    assert faithful["optimization"]["expected_per_device_batch_size"] == 192
    assert matched["model"]["L_cycles"] == 4
    assert matched["model"]["pos_encodings"] == "rope"
    assert matched["seed"] == 42
    assert matched["optimization"]["global_batch_size"] == 32
    assert matched["optimization"]["expected_per_device_batch_size"] == 8
    assert matched["optimization"]["gradient_accumulation_steps"] == 1
    assert faithful["epochs"] == matched["epochs"] == 100

    plan = _validate_config(matched, world_size=4)
    assert plan["per_device_batch_size"] == 8
    assert plan["microbatch_global_size"] == 32
    assert plan["gradient_accumulation_steps"] == 1
    assert plan["effective_global_batch_size"] == 32

    with pytest.raises(ValueError, match="expects world_size=4"):
        _validate_config(matched, world_size=1)

    mismatched = {**matched, "optimization": dict(matched["optimization"])}
    mismatched["optimization"]["global_batch_size"] = 768
    with pytest.raises(ValueError, match="resolves to 192 examples per rank"):
        _validate_config(mismatched, world_size=4)

    smaller = load_experiment_config(
        PROJECT_ROOT / "configs" / "trm_matched_mamba2_mlp_100_epochs_bs4.json"
    )
    smaller_plan = _validate_config(smaller, world_size=4)
    assert smaller_plan["per_device_batch_size"] == 4
    assert smaller_plan["effective_global_batch_size"] == 32
