from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import json
import numpy as np

from arc_agi_trm.experiment import (
    batch_sizes,
    exact_task_metrics,
    learning_rate_for_step,
    load_experiment_config,
    seed_training_rank,
    train_one_epoch,
    validation_improved,
)
from arc_agi_trm.checkpoints import load_checkpoint
from arc_agi_trm.runner import (
    create_hybrid_model,
    create_loaders,
    create_optimizers,
    evaluate_validation,
)
from scripts.build_arc_agi2_from_kaggle import build_dataset
from scripts.train_mamba2_hybrid import _init_distributed, _load_resume, _save_checkpoint
from test_mamba2_hybrid import FakeMamba2
from test_trm_smoke import synthetic_batch, tiny_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_experiment_config():
    model_config = tiny_config()
    model_config.update(mamba_d_state=4, mamba_headdim=8, mamba_expand=1, mamba_d_conv=4)
    return {
        "operator": "mamba2_attention",
        "seed": 11,
        "epochs": 2,
        "data": {"num_aug": 0, "validation_tasks": 1},
        "model": model_config,
        "optimization": {
            "global_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "lr": 1e-3,
            "puzzle_emb_lr": 1e-3,
            "lr_min_ratio": 1.0,
            "lr_warmup_steps": 2,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.0,
            "puzzle_emb_weight_decay": 0.0,
            "gradient_clip_norm": None,
        },
        "run": {"eval_every_epochs": 1, "checkpoint_every_epochs": 1, "eval_attempts": 1, "num_workers": 0},
        "estimated_total_optimizer_steps": 4,
    }


def _model_and_optimizers(config=None):
    config = config or _small_experiment_config()
    metadata = SimpleNamespace(seq_len=9, vocab_size=12, num_puzzle_identifiers=4)
    model = create_hybrid_model(
        config,
        metadata,
        per_device_batch_size=2,
        device=torch.device("cpu"),
        mamba_cls=FakeMamba2,
    )
    optimizers, base_lrs = create_optimizers(
        model,
        config,
        world_size=1,
        dense_optimizer_cls=torch.optim.AdamW,
    )
    return model, optimizers, base_lrs


def test_both_experiment_presets_only_change_budget_and_run_identity() -> None:
    short = load_experiment_config(PROJECT_ROOT / "configs" / "mamba2_hybrid_100_epochs.json")
    long = load_experiment_config(PROJECT_ROOT / "configs" / "mamba2_hybrid_1000_epochs.json")
    assert short["epochs"] == 100
    assert long["epochs"] == 1000
    for key in ("operator", "seed", "data", "model", "optimization", "run"):
        assert short[key] == long[key]
    assert short["model"]["hidden_size"] == 512
    assert short["model"]["mamba_d_state"] == 128
    assert short["model"]["mamba_d_conv"] == 4
    assert short["optimization"]["global_batch_size"] == 32
    assert short["data"]["num_aug"] == 1000


def test_batch_sizes_reject_nondivisible_distributed_global_batch() -> None:
    with pytest.raises(ValueError, match="divisible"):
        batch_sizes(30, 4, 1)


def test_distributed_sampler_numpy_stream_matches_across_ranks() -> None:
    seed_training_rank(27, 0)
    rank0_numpy = np.random.randint(0, 100000, size=8)
    rank0_torch = torch.rand(4)
    seed_training_rank(27, 1)
    rank1_numpy = np.random.randint(0, 100000, size=8)
    rank1_torch = torch.rand(4)
    assert np.array_equal(rank0_numpy, rank1_numpy)
    assert not torch.equal(rank0_torch, rank1_torch)


def test_tiny_synthetic_dataset_runs_optimizer_steps_with_hybrid_model() -> None:
    config = _small_experiment_config()
    model, optimizers, base_lrs = _model_and_optimizers(config)
    batch = synthetic_batch()
    result = train_one_epoch(
        model=model,
        loader=[("all", batch, 2), ("all", batch, 2)],
        optimizers=optimizers,
        optimizer_base_lrs=base_lrs,
        device=torch.device("cpu"),
        carry=None,
        optimizer_step=0,
        total_steps=4,
        warmup_steps=2,
        min_lr_ratio=1.0,
        global_batch_size=2,
    )
    assert result.optimizer_step == 2
    assert result.examples_sampled == 4
    assert result.local_rows_forwarded == 4
    assert torch.isfinite(torch.tensor(result.loss_sum_local))
    assert len(result.learning_rates) == 2
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_accumulation_flushes_a_short_final_window() -> None:
    config = _small_experiment_config()
    model, optimizers, base_lrs = _model_and_optimizers(config)
    batch = synthetic_batch()
    result = train_one_epoch(
        model=model,
        loader=[("all", batch, 2), ("all", batch, 2), ("all", batch, 2)],
        optimizers=optimizers,
        optimizer_base_lrs=base_lrs,
        device=torch.device("cpu"),
        carry=None,
        optimizer_step=0,
        total_steps=4,
        warmup_steps=2,
        min_lr_ratio=1.0,
        global_batch_size=2,
        accumulation_steps=2,
    )
    assert result.optimizer_step == 2
    assert result.examples_sampled == 6
    assert len(result.learning_rates) == 2


def test_gradient_accumulation_preserves_sparse_embedding_ids() -> None:
    from models.sparse_embedding import (
        CastedSparseEmbedding,
        CastedSparseEmbeddingSignSGD_Distributed,
    )

    class SparseToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = CastedSparseEmbedding(
                num_embeddings=4,
                embedding_dim=2,
                batch_size=2,
                init_std=0,
                cast_to=torch.float32,
            )

        def initial_carry(self, batch):
            return None

        def forward(self, *, carry, batch, return_keys):
            embedded = self.embedding(batch["ids"])
            loss = ((embedded - batch["target"]) ** 2).sum()
            return carry, loss, {}, None, torch.tensor(True)

    model = SparseToy()
    sparse_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
        model.embedding.buffers(), lr=0.0, weight_decay=0.0, world_size=1
    )
    batches = [
        (
            "all",
            {
                "ids": torch.tensor([0, 1]),
                "inputs": torch.zeros(2, 1),
                "target": torch.ones(2, 2),
            },
            2,
        ),
        (
            "all",
            {
                "ids": torch.tensor([1, 2]),
                "inputs": torch.zeros(2, 1),
                "target": torch.full((2, 2), 2.0),
            },
            2,
        ),
    ]
    result = train_one_epoch(
        model=model,
        loader=batches,
        optimizers=[sparse_optimizer],
        optimizer_base_lrs=[0.1],
        device=torch.device("cpu"),
        carry=None,
        optimizer_step=0,
        total_steps=1,
        warmup_steps=0,
        min_lr_ratio=1.0,
        global_batch_size=2,
        accumulation_steps=2,
    )
    assert result.optimizer_step == 1
    assert torch.equal(
        model.embedding.weights[:3], torch.full((3, 2), 0.1)
    )


def test_checkpoint_roundtrip_restores_optimizer_rng_carry_and_epoch(tmp_path) -> None:
    config = _small_experiment_config()
    config["estimated_total_optimizer_steps"] = 8
    model, optimizers, _ = _model_and_optimizers(config)
    batch = synthetic_batch()
    trained = train_one_epoch(
        model=model,
        loader=[("all", batch, 2)],
        optimizers=optimizers,
        optimizer_base_lrs=[1e-3, 1e-3],
        device=torch.device("cpu"),
        carry=None,
        optimizer_step=6,
        total_steps=8,
        warmup_steps=2,
        min_lr_ratio=1.0,
        global_batch_size=2,
    )
    class DatasetState:
        _iters = 2

    dataset = DatasetState()
    best = {"validation_task_exact_match": 0.25, "validation_loss": 1.2}
    _save_checkpoint(
        run_dir=tmp_path,
        model=model,
        optimizers=optimizers,
        config=config,
        epoch=2,
        optimizer_step=trained.optimizer_step,
        best_metrics=best,
        train_dataset=dataset,
        carry=trained.carry,
        rank=0,
        world_size=1,
        cpu_group=None,
        timing_summary={"epoch_duration_sum": 12.5, "epoch_duration_count": 2},
        is_best=True,
    )
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best.pt").is_file()

    saved = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    expected_rng = saved["rank_runtime"][0]["rng"]["torch_cpu"]
    assert saved["scheduler"]["step"] == 7
    assert saved["optimizers"][1]["state"]
    restored_model, restored_optimizers, _ = _model_and_optimizers(config)
    restored_dataset = DatasetState()
    restored_dataset._iters = 0
    epoch, step, restored_carry, restored_best, restored_timing = _load_resume(
        path=tmp_path / "latest.pt",
        model=restored_model,
        optimizers=restored_optimizers,
        config=config,
        train_dataset=restored_dataset,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    assert (epoch, step, restored_dataset._iters) == (2, 7, 2)
    assert restored_best == best
    assert restored_timing == {"epoch_duration_sum": 12.5, "epoch_duration_count": 2}
    assert restored_carry.inner_carry.z_H.shape == trained.carry.inner_carry.z_H.shape
    assert torch.equal(restored_carry.current_data["inputs"], batch["inputs"])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, dict(restored_model.named_parameters())[name])
    restored_dense_state = restored_optimizers[1].state_dict()["state"]
    assert restored_dense_state.keys() == saved["optimizers"][1]["state"].keys()
    for parameter_id, parameter_state in restored_dense_state.items():
        for key, value in parameter_state.items():
            saved_value = saved["optimizers"][1]["state"][parameter_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value.cpu(), saved_value)
            else:
                assert value == saved_value

    _save_checkpoint(
        run_dir=tmp_path,
        model=model,
        optimizers=optimizers,
        config=config,
        epoch=3,
        optimizer_step=trained.optimizer_step,
        best_metrics=best,
        train_dataset=dataset,
        carry=trained.carry,
        rank=0,
        world_size=1,
        cpu_group=None,
        timing_summary={"epoch_duration_sum": 13.0, "epoch_duration_count": 3},
        is_best=False,
    )
    latest = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    best_saved = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert latest["completed_epochs"] == 3
    assert best_saved["completed_epochs"] == 2
    for key, value in saved["model"].items():
        assert torch.equal(best_saved["model"][key], value)

    incomplete_path = tmp_path / "incomplete.pt"
    incomplete = dict(latest)
    incomplete.pop("optimizers")
    torch.save(incomplete, incomplete_path)
    untouched_model, untouched_optimizers, _ = _model_and_optimizers(config)
    with pytest.raises(ValueError, match="missing required state: optimizers"):
        _load_resume(
            path=incomplete_path,
            model=untouched_model,
            optimizers=untouched_optimizers,
            config=config,
            train_dataset=DatasetState(),
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )


def test_best_model_selection_maximizes_task_accuracy_then_minimizes_loss() -> None:
    best = {"validation_task_exact_match": 0.5, "validation_loss": 1.0}
    assert validation_improved(
        {"validation_task_exact_match": 0.75, "validation_loss": 1.4}, best
    )
    assert validation_improved(
        {"validation_task_exact_match": 0.5, "validation_loss": 0.9}, best
    )
    assert not validation_improved(
        {"validation_task_exact_match": 0.25, "validation_loss": 0.1}, best
    )
    assert not validation_improved(
        {"validation_task_exact_match": 0.5, "validation_loss": 1.0}, best
    )


def test_corrupted_full_checkpoint_fails_clearly(tmp_path) -> None:
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a pytorch checkpoint")
    with pytest.raises(ValueError, match="Could not read trusted training checkpoint"):
        load_checkpoint(path, trusted=True)


def test_task_metric_requires_every_test_input_to_match() -> None:
    expected = {"a": [("i1", "o1"), ("i2", "o2")], "b": [("i3", "o3")]}
    predicted = [
        ("a", "i1", "o1"),
        ("a", "i2", "wrong"),
        ("b", "i3", "o3"),
    ]
    metrics = exact_task_metrics(expected, predicted)
    assert metrics["tasks_evaluated"] == 2
    assert metrics["tasks_solved"] == 1
    assert metrics["task_exact_match"] == 0.5
    assert metrics["grid_exact_accuracy"] == pytest.approx(2 / 3)


def test_validation_evaluates_original_task_test_pairs_and_reports_task_metrics(tmp_path) -> None:
    source = tmp_path / "competition"
    source.mkdir()
    challenges = {
        "train-a": {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[2]]}]},
        "valid-b": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]},
    }
    solutions = {"train-a": [[[3]]], "valid-b": [[[4]]]}
    (source / "arc-agi_training_challenges.json").write_text(json.dumps(challenges), encoding="utf-8")
    (source / "arc-agi_training_solutions.json").write_text(json.dumps(solutions), encoding="utf-8")
    dataset_dir = tmp_path / "dataset"
    build_dataset(
        source,
        dataset_dir,
        num_aug=0,
        seed=9,
        validation_tasks=1,
        expected_task_count=2,
    )
    _train_loader, _train_dataset, validation_loader, validation_dataset = create_loaders(
        dataset_dir,
        seed=9,
        global_batch_size=2,
        rank=0,
        world_size=1,
    )

    model_config = {
        "batch_size": 2,
        "seq_len": 900,
        "puzzle_emb_ndim": 16,
        "num_puzzle_identifiers": validation_dataset.metadata.num_puzzle_identifiers,
        "vocab_size": 12,
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
        "mamba_d_conv": 4,
    }
    config = {"operator": "mamba2_attention", "model": model_config}
    model = create_hybrid_model(
        config,
        validation_dataset.metadata,
        per_device_batch_size=2,
        device=torch.device("cpu"),
        mamba_cls=FakeMamba2,
    )
    identifiers = json.loads((dataset_dir / "identifiers.json").read_text(encoding="utf-8"))
    test_puzzles = json.loads((dataset_dir / "test_puzzles.json").read_text(encoding="utf-8"))
    metrics = evaluate_validation(
        loss_model=model,
        loader=validation_loader,
        identifiers=identifiers,
        test_puzzles=test_puzzles,
        device=torch.device("cpu"),
    )
    assert metrics is not None
    assert metrics["validation_tasks_evaluated"] == 1
    assert metrics["validation_grid_examples"] == 1
    assert metrics["validation_attempts"] == 1
    assert torch.isfinite(torch.tensor(metrics["validation_loss"]))


def test_scheduler_uses_saved_optimizer_step() -> None:
    assert learning_rate_for_step(
        base_lr=1.0, step=1, warmup_steps=4, total_steps=20, min_ratio=1.0
    ) == 0.25
    assert learning_rate_for_step(
        base_lr=1.0, step=12, warmup_steps=4, total_steps=20, min_ratio=1.0
    ) == 1.0


def test_training_command_fails_clearly_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        _init_distributed()

