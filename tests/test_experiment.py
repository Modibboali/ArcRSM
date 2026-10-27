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
from arc_agi_trm.numerics import NumericsDiagnostics
from arc_agi_trm.runner import (
    create_hybrid_model,
    create_loaders,
    create_optimizers,
    evaluate_validation,
)
from scripts.build_arc_agi2_from_kaggle import build_dataset
from scripts.train_mamba2_hybrid import (
    _init_distributed,
    _install_numerics_hooks,
    _load_resume,
    _save_checkpoint,
)
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


def test_mamba2_mlp_depth_ablation_presets_only_change_requested_depth() -> None:
    two_layer = load_experiment_config(PROJECT_ROOT / "configs" / "mamba2_mlp_100_epochs.json")
    one_layer = load_experiment_config(PROJECT_ROOT / "configs" / "mamba2_mlp_depth1_100_epochs.json")
    assert two_layer["model"]["L_layers"] == 2
    assert one_layer["model"]["L_layers"] == 1
    for key in ("operator", "seed", "epochs", "data", "optimization", "run"):
        assert two_layer[key] == one_layer[key]
    assert {key: value for key, value in two_layer["model"].items() if key != "L_layers"} == {
        key: value for key, value in one_layer["model"].items() if key != "L_layers"
    }
    lcycle2_two = load_experiment_config(PROJECT_ROOT / "configs" / "mamba2_mlp_lcycles2_100_epochs.json")
    lcycle2_one = load_experiment_config(PROJECT_ROOT / "configs" / "mamba2_mlp_depth1_lcycles2_100_epochs.json")
    assert lcycle2_two["model"]["L_cycles"] == lcycle2_one["model"]["L_cycles"] == 2
    assert lcycle2_two["model"]["L_layers"] == 2
    assert lcycle2_one["model"]["L_layers"] == 1


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

    import copy

    bad_model = copy.deepcopy(latest)
    bad_model_key = next(
        key for key, value in bad_model["model"].items() if value.is_floating_point()
    )
    bad_model["model"][bad_model_key].view(-1)[0] = float("nan")
    bad_model_path = tmp_path / "nonfinite-model.pt"
    torch.save(bad_model, bad_model_path)
    untouched_model, untouched_optimizers, _ = _model_and_optimizers(config)
    untouched_parameter = next(untouched_model.parameters()).detach().clone()
    with pytest.raises(ValueError, match="non-finite model state"):
        _load_resume(
            path=bad_model_path,
            model=untouched_model,
            optimizers=untouched_optimizers,
            config=config,
            train_dataset=DatasetState(),
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )
    assert torch.equal(next(untouched_model.parameters()), untouched_parameter)

    bad_optimizer = copy.deepcopy(latest)
    state_tensor = next(
        tensor
        for state in bad_optimizer["optimizers"][1]["state"].values()
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
    )
    state_tensor.view(-1)[0] = float("inf")
    bad_optimizer_path = tmp_path / "nonfinite-optimizer.pt"
    torch.save(bad_optimizer, bad_optimizer_path)
    with pytest.raises(ValueError, match="non-finite optimizer state"):
        _load_resume(
            path=bad_optimizer_path,
            model=untouched_model,
            optimizers=untouched_optimizers,
            config=config,
            train_dataset=DatasetState(),
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )
    assert torch.equal(next(untouched_model.parameters()), untouched_parameter)

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


def test_numerics_diagnostic_logs_first_bad_tensor_and_batch_identifiers(tmp_path) -> None:
    log_path = tmp_path / "numerics_rank0.jsonl"
    diagnostics = NumericsDiagnostics(
        rank=0,
        world_size=1,
        log_path=log_path,
        puzzle_names=["<blank>", "task-a", "task-b"],
    )
    diagnostics.set_context(
        epoch=7,
        optimizer_step=799,
        batch={"puzzle_identifiers": torch.tensor([1, 2, 1])},
    )
    diagnostics.check_tensor(
        "forward.lm_head.logits",
        torch.tensor([[0.0, float("inf")]], dtype=torch.bfloat16),
        stage="forward",
    )
    with pytest.raises(FloatingPointError, match="lm_head.logits"):
        diagnostics.synchronize("forward", device=torch.device("cpu"))

    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["rank"] == 0
    assert record["epoch"] == 7
    assert record["optimizer_step"] == 799
    assert record["tensor_name"] == "forward.lm_head.logits"
    assert record["shape"] == [1, 2]
    assert record["dtype"] == "torch.bfloat16"
    assert record["non_finite_count"] == 1
    assert record["puzzle_identifiers"] == [1, 2]
    assert record["puzzle_names"] == ["task-a", "task-b"]
    stats = NumericsDiagnostics.tensor_stats(torch.tensor([3.0, 4.0, float("inf")]))
    assert stats["finite_count"] == 2
    assert stats["non_finite_count"] == 1
    assert stats["l2_norm"] == pytest.approx(5.0)
    assert stats["std"] == pytest.approx(0.5)


def test_recursion_diagnostic_labels_shared_invocations_and_captures_gradients(
    tmp_path, capsys
) -> None:
    diagnostics = NumericsDiagnostics(
        rank=0,
        world_size=1,
        log_path=tmp_path / "numerics_rank0.jsonl",
    )
    diagnostics.set_context(epoch=2, optimizer_step=17)
    diagnostics.begin_recursion_forward(h_cycles=3, l_cycles=4, l_layers=1)
    loss = torch.tensor(0.0)
    for call_index in range(15):
        has_grad = call_index >= 10
        with torch.set_grad_enabled(has_grad):
            diagnostics.begin_l_level_call()
            hidden = torch.tensor([1.0, 2.0], requires_grad=True)
            module_path = "model.inner.L_level.layers.0"
            diagnostics.start_block_invocation(module_path, 0, hidden)
            normalized = hidden / 2
            branch = normalized * 0.25
            residual = hidden + branch
            diagnostics.record_invocation_tensor(module_path, "mamba_norm_input", normalized)
            diagnostics.record_invocation_tensor(module_path, "mamba_output", branch)
            diagnostics.record_invocation_tensor(module_path, "mamba_residual", residual)
            output = residual + normalized
            diagnostics.end_block_invocation(module_path, output)
            if has_grad:
                loss = loss + output.sum()
    loss.backward()
    diagnostics.finish_recursion_forward()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    records = [
        json.loads(line)
        for line in (tmp_path / "propagation_rank0.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    invocations = records[0]["invocations"]
    assert len(invocations) == 15
    assert invocations[0]["h_cycle"] == 0
    assert invocations[0]["latent_update"] == "L0"
    assert invocations[0]["autograd_enabled"] is False
    assert invocations[10]["h_cycle"] == 2
    assert invocations[10]["latent_update"] == "L0"
    assert invocations[14]["latent_update"] == "H"
    assert invocations[10]["gradient"]["block_input"]["l2_norm"] > 0
    assert invocations[10]["forward"]["mamba_residual"]["finite_count"] == 2


def test_mamba2_mlp_hooks_capture_actual_shared_layer_calls(tmp_path) -> None:
    config = _small_experiment_config()
    config["operator"] = "mamba2_mlp"
    config["model"].update(
        mamba_d_ssm=None,
        mamba_ngroups=1,
        mamba_rmsnorm=True,
        mamba_norm_before_gate=False,
        mamba_chunk_size=16,
        mamba_use_mem_eff_path=True,
        mamba_bias=False,
        mamba_conv_bias=True,
    )
    model, optimizers, base_lrs = _model_and_optimizers(config)
    batch = synthetic_batch()
    diagnostics = NumericsDiagnostics(
        rank=0,
        world_size=1,
        log_path=tmp_path / "numerics_rank0.jsonl",
    )
    _install_numerics_hooks(model, diagnostics)
    train_one_epoch(
        model=model,
        loader=[("all", batch, 2)],
        optimizers=optimizers,
        optimizer_base_lrs=base_lrs,
        device=torch.device("cpu"),
        carry=None,
        optimizer_step=0,
        total_steps=1,
        warmup_steps=0,
        min_lr_ratio=1.0,
        global_batch_size=2,
        diagnostics=diagnostics,
        epoch=2,
        vocab_size=12,
        num_puzzle_identifiers=4,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "propagation_rank0.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    record = next(item for item in records if item["event"] == "recursion_forward_summary")
    invocations = record["invocations"]
    assert len(invocations) == 3 * (2 + 1)
    assert invocations[0]["module"].endswith("L_level.layers.0")
    assert invocations[0]["autograd_enabled"] is False
    assert invocations[6]["h_cycle"] == 2
    assert invocations[6]["latent_update"] == "L0"
    assert invocations[6]["autograd_enabled"] is True
    assert invocations[6]["forward"]["mamba_output"]["finite_count"] > 0
    assert invocations[6]["gradient"]["mlp_output"]["l2_norm"] > 0
    gradient_summary = next(
        item for item in records
        if item["event"] == "gradient_summary"
        and item["phase"] == "immediately_before_optimizer_update"
    )
    assert gradient_summary["q_head_weight_name"].endswith("inner.q_head.weight")
    assert gradient_summary["q_head_weight_gradient"]["l2_norm"] > 0
    q_head_state = [item for item in records if item["event"] == "q_head_optimizer_state"]
    after_step = next(
        item for item in q_head_state
        if item["phase"] == "immediately_after_optimizer_step"
    )
    assert after_step["optimizer_class"] == "AdamW"
    assert after_step["fields"]["exp_avg_sq"]["finite_count"] == 2 * 32


def test_train_diagnostics_stop_before_backward_on_nonfinite_loss(tmp_path) -> None:
    class NonFiniteLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def initial_carry(self, _batch):
            return None

        def forward(self, *, carry, batch, return_keys):
            loss = self.weight * batch["inputs"].float().sum() * float("nan")
            return carry, loss, {"lm_loss": loss.detach()}, None, torch.tensor(True)

    model = NonFiniteLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    diagnostics = NumericsDiagnostics(
        rank=0,
        world_size=1,
        log_path=tmp_path / "train-numerics.jsonl",
    )
    batch = {
        "inputs": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[2, 3]]),
        "puzzle_identifiers": torch.tensor([1]),
    }
    with pytest.raises(FloatingPointError, match="loss_component.lm_loss"):
        train_one_epoch(
            model=model,
            loader=[("all", batch, 1)],
            optimizers=[optimizer],
            optimizer_base_lrs=[0.1],
            device=torch.device("cpu"),
            carry=None,
            optimizer_step=0,
            total_steps=1,
            warmup_steps=0,
            min_lr_ratio=1.0,
            global_batch_size=1,
            diagnostics=diagnostics,
            epoch=7,
            vocab_size=12,
            num_puzzle_identifiers=2,
        )
    assert model.weight.grad is None
    record = json.loads(
        (tmp_path / "train-numerics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["epoch"] == 7
    assert record["optimizer_step"] == 0
    assert record["tensor_name"] == "loss_component.lm_loss"


def test_checkpoint_save_rejects_nonfinite_state_without_replacing_latest(tmp_path) -> None:
    config = _small_experiment_config()
    model, optimizers, _ = _model_and_optimizers(config)
    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"preserve-me")
    with torch.no_grad():
        next(model.parameters()).fill_(float("inf"))

    class DatasetState:
        _iters = 0

    with pytest.raises(FloatingPointError, match="checkpoint save"):
        _save_checkpoint(
            run_dir=tmp_path,
            model=model,
            optimizers=optimizers,
            config=config,
            epoch=1,
            optimizer_step=1,
            best_metrics={},
            train_dataset=DatasetState(),
            carry=None,
            rank=0,
            world_size=1,
            cpu_group=None,
            timing_summary={"epoch_duration_sum": 0.0, "epoch_duration_count": 0},
            is_best=False,
        )
    assert latest.read_bytes() == b"preserve-me"


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
