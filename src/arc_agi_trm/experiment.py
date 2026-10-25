"""Reusable configuration, state, and training helpers for ARC experiments."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

from .numerics import NumericsDiagnostics


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a run preset layered over the shared Mamba-2 base JSON file."""

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        selected = json.load(stream)
    base_path = selected.pop("base", None)
    if base_path is None:
        return selected
    with (path.parent / base_path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    for key, value in selected.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def batch_sizes(global_batch_size: int, world_size: int, accumulation_steps: int) -> dict[str, int]:
    if global_batch_size <= 0 or world_size <= 0 or accumulation_steps <= 0:
        raise ValueError("batch size, world size, and accumulation steps must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by world_size={world_size}"
        )
    return {
        "per_device_batch_size": global_batch_size // world_size,
        "microbatch_global_size": global_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_global_batch_size": global_batch_size * accumulation_steps,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_training_rank(seed: int, rank: int) -> None:
    """Seed rank-local model RNGs while keeping the upstream sampler in sync.

    ``PuzzleDataset._sample_batch`` uses NumPy's process-global RNG to choose
    examples, then slices the same global batch by rank. NumPy therefore needs
    the same stream on every rank; Python and torch RNGs can be rank-local.
    """

    random.seed(seed + rank)
    np.random.seed(seed)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def tree_map(value: Any, tensor_fn) -> Any:
    """Map tensors in nested state, including the upstream TRM carry dataclasses."""

    if isinstance(value, torch.Tensor):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {key: tree_map(item, tensor_fn) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_map(item, tensor_fn) for item in value]
    if isinstance(value, tuple):
        return type(value)(*(tree_map(item, tensor_fn) for item in value)) if hasattr(value, "_fields") else tuple(
            tree_map(item, tensor_fn) for item in value
        )
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        return type(value)(**{
            name: tree_map(getattr(value, name), tensor_fn) for name in fields
        })
    return value


def checkpoint_fingerprint(config: Mapping[str, Any]) -> str:
    stable = {
        key: config[key]
        for key in ("operator", "seed", "epochs", "data", "model", "optimization", "run")
    }
    stable["model"] = dict(stable["model"])
    # The local sparse-embedding buffers use batch size, but it is not a learned
    # model shape and can differ between training and standalone evaluation.
    stable["model"].pop("batch_size", None)
    stable["dataset_manifest_fingerprint"] = config.get("dataset_manifest_fingerprint")
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def learning_rate_for_step(
    *, base_lr: float, step: int, warmup_steps: int, total_steps: int, min_ratio: float
) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(base_lr * (min_ratio + max(0.0, (1.0 - min_ratio) * cosine)))


def assert_finite_tensor(tensor: torch.Tensor, description: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values detected in {description}")


@dataclass
class EpochResult:
    carry: Any
    optimizer_step: int
    loss_sum_local: float
    examples_sampled: int
    local_rows_forwarded: int
    metric_sums_local: dict[str, float]
    learning_rates: list[float]


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: Iterable,
    optimizers: Sequence[torch.optim.Optimizer],
    optimizer_base_lrs: Sequence[float],
    device: torch.device,
    carry: Any,
    optimizer_step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    global_batch_size: int,
    accumulation_steps: int = 1,
    gradient_clip_norm: float | None = None,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
    diagnostics: NumericsDiagnostics | None = None,
    vocab_size: int | None = None,
    num_puzzle_identifiers: int | None = None,
) -> EpochResult:
    """Run one sampler epoch with upstream ACT carries and loss semantics.

    The raw per-example loss is accumulated. Dense gradients are summed across
    ranks then divided by the number of examples in the accumulation window.
    Sparse puzzle-embedding gradients and their matching IDs are gathered per
    microbatch, then passed through the upstream SignSGD update once per window.
    """

    if len(optimizers) != len(optimizer_base_lrs):
        raise ValueError("optimizers and optimizer_base_lrs must have equal lengths")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    loss_sum_local = 0.0
    examples_sampled = 0
    local_rows_forwarded = 0
    metric_sums: dict[str, float] = {}
    used_lrs: list[float] = []
    accumulated_global_examples = 0
    accumulated_microbatches = 0
    pending = False
    sparse_accumulated: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def sparse_embedding_parts(optimizer):
        if optimizer.__class__.__name__ != "CastedSparseEmbeddingSignSGD_Distributed":
            return None
        if len(optimizer.param_groups) != 1 or len(optimizer.param_groups[0]["params"]) != 3:
            raise ValueError("Unexpected upstream sparse SignSGD parameter layout")
        weights, local_weights, local_ids = optimizer.param_groups[0]["params"]
        return weights, local_weights, local_ids

    sparse_parts = {}
    for optimizer in optimizers:
        parts = sparse_embedding_parts(optimizer)
        if parts is not None:
            sparse_parts[id(optimizer)] = (optimizer, parts)

    parameter_names = {}
    if diagnostics is not None:
        parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        parameter_names.update(
            {
                id(buffer): name
                for name, buffer in model.named_buffers()
                if id(buffer) not in parameter_names
            }
        )

    def check_parameters_and_optimizer_state(stage: str) -> None:
        if diagnostics is None:
            return
        for name, parameter in model.named_parameters():
            diagnostics.check_tensor(f"model.parameter.{name}", parameter, stage=stage)
        for name, buffer in model.named_buffers():
            diagnostics.check_tensor(f"model.buffer.{name}", buffer, stage=stage)
        for optimizer_index, optimizer in enumerate(optimizers):
            for parameter, state in optimizer.state.items():
                parameter_name = parameter_names.get(id(parameter), "unnamed_parameter")
                diagnostics.check_tree(
                    f"optimizer[{optimizer_index}].state.{parameter_name}",
                    state,
                    stage=stage,
                )
                if diagnostics.first_failure is not None:
                    return

    def check_gradients(stage: str) -> None:
        if diagnostics is None:
            return
        seen: set[int] = set()
        for optimizer_index, optimizer in enumerate(optimizers):
            for group_index, group in enumerate(optimizer.param_groups):
                for parameter_index, parameter in enumerate(group["params"]):
                    if id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    if parameter.grad is not None:
                        name = parameter_names.get(
                            id(parameter),
                            f"optimizer[{optimizer_index}].group[{group_index}].parameter[{parameter_index}]",
                        )
                        diagnostics.check_tensor(
                            f"gradient.{name}", parameter.grad, stage=stage
                        )
                        if diagnostics.first_failure is not None:
                            return

    def optimizer_update() -> None:
        nonlocal optimizer_step, accumulated_global_examples, accumulated_microbatches, pending
        if not pending or accumulated_global_examples <= 0:
            return
        if diagnostics is not None:
            check_gradients("before_optimizer_update")
            check_parameters_and_optimizer_state("before_optimizer_update")
            diagnostics.synchronize("before_optimizer_update", device=device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                if diagnostics is None:
                    assert_finite_tensor(parameter.grad, "model gradient")
                if world_size > 1:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(accumulated_global_examples)

        if diagnostics is not None:
            check_gradients("after_distributed_gradient_reduction")
            diagnostics.synchronize("after_distributed_gradient_reduction", device=device)

        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None and diagnostics is None:
                        assert_finite_tensor(parameter.grad, "optimizer parameter gradient")

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if diagnostics is not None:
                check_gradients("after_gradient_clipping")
                diagnostics.synchronize("after_gradient_clipping", device=device)

        optimizer_step += 1
        last_lr = None
        for optimizer, base_lr in zip(optimizers, optimizer_base_lrs):
            lr = learning_rate_for_step(
                base_lr=base_lr,
                step=optimizer_step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_ratio=min_lr_ratio,
            )
            last_lr = lr
            for group in optimizer.param_groups:
                group["lr"] = lr
            sparse_entry = sparse_parts.get(id(optimizer))
            if sparse_entry is not None and accumulation_steps > 1:
                _sparse_optimizer, parts = sparse_entry
                weights, _local_weights, _local_ids = parts
                batches = sparse_accumulated.pop(id(optimizer), [])
                if batches:
                    local_ids = torch.cat([ids for ids, _grad in batches], dim=0)
                    local_grad = torch.cat([grad for _ids, grad in batches], dim=0)
                    from models.sparse_embedding import _sparse_emb_signsgd_dist

                    group = optimizer.param_groups[0]
                    _sparse_emb_signsgd_dist(
                        local_grad,
                        local_ids,
                        weights,
                        lr=float(group["lr"]),
                        weight_decay=float(group["weight_decay"]),
                        world_size=int(group["world_size"]),
                    )
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if last_lr is not None:
            used_lrs.append(last_lr)

        if diagnostics is not None:
            diagnostics.set_context(epoch=epoch, optimizer_step=optimizer_step)
            check_parameters_and_optimizer_state("after_optimizer_update")
            diagnostics.synchronize("after_optimizer_update", device=device)

        accumulated_global_examples = 0
        accumulated_microbatches = 0
        pending = False

    if diagnostics is not None:
        initial_batch = getattr(carry, "current_data", None)
        diagnostics.set_context(
            epoch=epoch,
            optimizer_step=optimizer_step,
            batch=initial_batch if isinstance(initial_batch, Mapping) else None,
        )
        diagnostics.check_tree("carry.input", carry, stage="epoch_start")
        check_parameters_and_optimizer_state("epoch_start")
        diagnostics.synchronize("epoch_start", device=device)

    for _set_name, source_batch, global_effective_size in loader:
        if global_effective_size <= 0:
            raise RuntimeError("Sampler emitted an empty training batch")
        batch = {key: value.to(device, non_blocking=True) for key, value in source_batch.items()}
        if diagnostics is not None:
            diagnostics.set_context(epoch=epoch, optimizer_step=optimizer_step, batch=batch)
            diagnostics.check_batch(
                batch,
                vocab_size=vocab_size if vocab_size is not None else 2**31 - 1,
                num_puzzle_identifiers=(
                    num_puzzle_identifiers
                    if num_puzzle_identifiers is not None
                    else 2**31 - 1
                ),
            )
            diagnostics.synchronize("batch_validation", device=device)
        if carry is None:
            with torch.device(device):
                carry = model.initial_carry(batch)
        carry, loss, metrics, _outputs, _all_finished = model(
            carry=carry,
            batch=batch,
            return_keys=(),
        )
        if diagnostics is None:
            assert_finite_tensor(loss, "training loss")
        else:
            diagnostics.check_tree("carry", carry, stage="forward")
            for key, value in metrics.items():
                diagnostics.check_tensor(f"loss_component.{key}", value, stage="forward")
            diagnostics.check_tensor("loss.total", loss, stage="forward")
            diagnostics.synchronize("forward", device=device)
            reduced_loss = loss.detach().to(torch.float64)
            if world_size > 1:
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss.div_(int(global_effective_size))
            diagnostics.check_tensor(
                "loss.reduced_global_mean", reduced_loss, stage="loss_reduction"
            )
            diagnostics.synchronize("loss_reduction", device=device)
        loss.backward()
        if diagnostics is not None:
            check_gradients("after_backward")
            diagnostics.synchronize("after_backward", device=device)
        if accumulation_steps > 1:
            for optimizer, parts in sparse_parts.values():
                _weights, local_weights, local_ids = parts
                if local_weights.grad is not None:
                    if diagnostics is None:
                        assert_finite_tensor(local_weights.grad, "sparse embedding gradient")
                    sparse_accumulated.setdefault(id(optimizer), []).append(
                        (local_ids.detach().clone(), local_weights.grad.detach().clone())
                    )
                    local_weights.grad = None
        loss_sum_local += float(loss.detach().float().item())
        examples_sampled += int(global_effective_size)
        local_rows_forwarded += int(batch["inputs"].shape[0])
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach().float().item())
        if diagnostics is not None:
            diagnostics.check_scalar("loss_sum_local", loss_sum_local, stage="batch_aggregation")
            for key, value in metric_sums.items():
                diagnostics.check_scalar(
                    f"metric_sum_local.{key}", value, stage="batch_aggregation"
                )
            diagnostics.synchronize("batch_aggregation", device=device)
        accumulated_global_examples += int(global_effective_size)
        accumulated_microbatches += 1
        pending = True

        if accumulated_microbatches >= accumulation_steps:
            optimizer_update()

    optimizer_update()
    if diagnostics is not None:
        diagnostics.set_context(epoch=epoch, optimizer_step=optimizer_step)
        diagnostics.check_scalar("loss_sum_local", loss_sum_local, stage="epoch_aggregation")
        diagnostics.synchronize("epoch_aggregation", device=device)
    return EpochResult(
        carry=carry,
        optimizer_step=optimizer_step,
        loss_sum_local=loss_sum_local,
        examples_sampled=examples_sampled,
        local_rows_forwarded=local_rows_forwarded,
        metric_sums_local=metric_sums,
        learning_rates=used_lrs,
    )


def exact_task_metrics(
    expected: Mapping[str, Sequence[tuple[str, str]]],
    predicted: Sequence[tuple[str, str, str]],
) -> dict[str, float | int]:
    """Compute grid and strict task exact match from input/output grid hashes."""

    from collections import Counter, defaultdict

    actual: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for task_id, input_hash, output_hash in predicted:
        actual[task_id][(input_hash, output_hash)] += 1

    tasks_solved = 0
    tasks_evaluated = 0
    total_grids = 0
    correct_grids = 0
    for task_id, pairs in expected.items():
        task_expected = Counter(pairs)
        task_actual = actual.get(task_id, Counter())
        if task_expected:
            tasks_evaluated += 1
            total_grids += sum(task_expected.values())
            correct_grids += sum(
                min(count, task_actual[pair]) for pair, count in task_expected.items()
            )
            if task_actual == task_expected:
                tasks_solved += 1
    return {
        "tasks_evaluated": tasks_evaluated,
        "tasks_solved": tasks_solved,
        "task_exact_match": tasks_solved / max(1, tasks_evaluated),
        "grid_exact_accuracy": correct_grids / max(1, total_grids),
        "grid_examples": total_grids,
        "grid_examples_solved": correct_grids,
    }


def validation_improved(candidate: Mapping[str, Any], best: Mapping[str, Any]) -> bool:
    """Select by maximum task exact match, then minimum validation loss."""

    if not best:
        return True
    candidate_score = float(candidate["validation_task_exact_match"])
    best_score = float(best.get("validation_task_exact_match", -1.0))
    candidate_loss = float(candidate["validation_loss"])
    best_loss = float(best.get("validation_loss", float("inf")))
    return candidate_score > best_score or (
        candidate_score == best_score and candidate_loss < best_loss
    )
