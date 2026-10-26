"""Train a project-owned TRM operator with the unchanged upstream TRM loop."""

from __future__ import annotations

import argparse
import importlib.metadata
from itertools import islice
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
import dataclasses
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arc_agi_trm.experiment import (  # noqa: E402
    batch_sizes,
    capture_rng_state,
    checkpoint_fingerprint,
    load_experiment_config,
    manifest_fingerprint,
    restore_rng_state,
    seed_everything,
    seed_training_rank,
    train_one_epoch,
    tree_map,
    validation_improved,
)
from arc_agi_trm.numerics import NumericsDiagnostics  # noqa: E402
from arc_agi_trm.checkpoints import (  # noqa: E402
    load_checkpoint as read_training_checkpoint,
    load_pretrained,
    save_pretrained,
    source_revision,
)
from arc_agi_trm.runner import (  # noqa: E402
    check_dataset_config,
    create_experiment_model,
    create_loaders,
    create_optimizers,
    estimate_train_optimizer_updates,
    evaluate_validation,
    load_dataset_manifest,
)
from arc_agi_trm.progress import TrainingProgress  # noqa: E402


def _resolve_from_root(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _init_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment entry point requires CUDA. No CUDA device is available; "
            "use CPU unit tests and run configured training in the prepared Kaggle image."
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        cpu_group = dist.new_group(backend="gloo") if world_size > 1 else None
    else:
        rank, world_size, cpu_group = 0, 1, None
    return rank, world_size, local_rank, cpu_group, torch.device("cuda", local_rank)


def _validate_config(config: dict[str, Any], world_size: int) -> dict[str, int]:
    operator = config.get("operator")
    if operator not in {"attention", "mamba2_attention", "mamba2_mlp"}:
        raise ValueError(
            "Experiment configuration must set operator to one of: "
            "attention, mamba2_attention, mamba2_mlp"
        )
    if int(config.get("epochs", 0)) <= 0:
        raise ValueError("Experiment epochs must be positive")
    if int(config["model"]["hidden_size"]) % int(config["model"]["num_heads"]):
        raise ValueError("hidden_size must be divisible by num_heads")
    if operator.startswith("mamba2_"):
        if int(config["model"]["hidden_size"]) % int(config["model"].get("mamba_headdim", 0) or 1):
            raise ValueError("hidden_size must be divisible by mamba_headdim")
        if operator == "mamba2_mlp":
            required = (
                "mamba_d_state", "mamba_headdim", "mamba_expand", "mamba_d_conv",
                "mamba_d_ssm", "mamba_ngroups", "mamba_rmsnorm",
                "mamba_norm_before_gate", "mamba_chunk_size",
                "mamba_use_mem_eff_path", "mamba_bias", "mamba_conv_bias",
            )
            missing = [key for key in required if key not in config["model"]]
            if missing:
                raise ValueError(
                    "mamba2_mlp config is missing required Mamba-2 setting(s): "
                    + ", ".join(missing)
                )
    if int(config["model"]["L_cycles"]) < 1 or int(config["model"]["H_cycles"]) < 1:
        raise ValueError("TRM recursive cycle counts must be positive")
    if int(config["model"]["L_layers"]) < 1:
        raise ValueError("The model requires at least one shared TRM block")
    expected_world_size = config.get("expected_world_size")
    if expected_world_size is not None and int(expected_world_size) != world_size:
        raise ValueError(
            f"This preset expects world_size={int(expected_world_size)}, "
            f"but torchrun resolved world_size={world_size}"
        )
    run_settings = config["run"]
    if int(run_settings.get("num_workers", 0)) != 0:
        raise ValueError(
            "Exact resume is configured with num_workers=0 so sampler RNG state can be saved"
        )
    if int(run_settings["eval_every_epochs"]) < 1 or int(run_settings["checkpoint_every_epochs"]) < 1:
        raise ValueError("Evaluation and checkpoint intervals must be positive")
    if int(run_settings["eval_attempts"]) != 1:
        raise ValueError("This preset uses one greedy top-1 validation attempt")
    plan = batch_sizes(
        int(config["optimization"]["global_batch_size"]),
        world_size,
        int(config["optimization"]["gradient_accumulation_steps"]),
    )
    expected_per_device = config["optimization"].get(
        "expected_per_device_batch_size"
    )
    if expected_per_device is not None and int(expected_per_device) != plan[
        "per_device_batch_size"
    ]:
        raise ValueError(
            "Batch configuration mismatch: global_batch_size is the aggregate "
            "microbatch across all ranks, so it resolves to "
            f"{plan['per_device_batch_size']} examples per rank at world_size={world_size}; "
            f"the preset explicitly expects {int(expected_per_device)}. "
            "Change global_batch_size and expected_per_device_batch_size together."
        )
    return plan


def _mamba_modules(layer):
    if hasattr(layer, "mamba"):
        return (layer.mamba,)
    if hasattr(layer, "mamba1") and hasattr(layer, "mamba2"):
        return (layer.mamba1, layer.mamba2)
    return ()


def _install_numerics_hooks(loss_model, diagnostics: NumericsDiagnostics) -> list[Any]:
    """Observe Mamba/MLP stages and label each shared-block recursion invocation."""

    from arc_agi_trm.operators import Mamba2MLPBlock

    handles = []
    model_config = loss_model.model.config
    for path, module in loss_model.named_modules():
        if path.endswith("inner.L_level"):
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, _args, _path=path: diagnostics.begin_l_level_call()
                )
            )

    for path, module in loss_model.named_modules():
        if isinstance(module, Mamba2MLPBlock):
            module._numerics_callback = (
                lambda operation, tensor, module_path=path: (
                    diagnostics.check_tensor(
                        f"forward.{module_path}.{operation}", tensor, stage="forward"
                    ),
                    diagnostics.record_invocation_tensor(module_path, operation, tensor),
                )
            )
            layer_match = re.search(r"(?:^|\.)layers\.(\d+)$", path)
            layer_index = int(layer_match.group(1)) if layer_match else None

            def observe_block_input(
                _module, args, kwargs, *, module_path=path, layer_index=layer_index
            ):
                hidden = kwargs.get("hidden_states")
                if hidden is None and args:
                    hidden = args[0]
                if isinstance(hidden, torch.Tensor):
                    diagnostics.check_tensor(
                        f"forward.{module_path}.block_input", hidden, stage="forward"
                    )
                    diagnostics.start_block_invocation(
                        module_path, layer_index, hidden
                    )

            handles.append(
                module.register_forward_pre_hook(observe_block_input, with_kwargs=True)
            )

            def observe_block_output(_module, _inputs, output, *, module_path=path):
                if isinstance(output, torch.Tensor):
                    diagnostics.end_block_invocation(module_path, output)

            handles.append(module.register_forward_hook(observe_block_output))
        leaf_name = path.rsplit(".", 1)[-1]
        inside_mamba_or_mlp = ".mamba." in f".{path}." or ".mlp." in f".{path}."
        watch_module = leaf_name in {
            "embed_tokens",
            "puzzle_emb",
            "L_level",
            "lm_head",
            "q_head",
            "mamba",
            "mlp",
        } or (inside_mamba_or_mlp and not any(module.children()))
        if not watch_module:
            continue

        def observe_output(_module, _inputs, output, *, module_path=path):
            diagnostics.check_tree(
                f"forward.{module_path}.output", output, stage="forward"
            )

        handles.append(module.register_forward_hook(observe_output))

    # This configuration is fixed for a run. A call to the shared L_level
    # corresponds to one latent L update or the following H update.
    diagnostics.recursion_shape = (
        int(model_config.H_cycles),
        int(model_config.L_cycles),
        int(model_config.L_layers),
    )
    return handles


def _atomic_torch_save(payload: Any, destination: Path) -> float:
    started = time.perf_counter()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return time.perf_counter() - started


def _first_nonfinite(value: Any, name: str = "state") -> dict[str, Any] | None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            bad_count = int((~torch.isfinite(value)).sum().detach().cpu().item())
            if bad_count:
                return {
                    "tensor_name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "non_finite_count": bad_count,
                }
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "tensor_name": name,
            "shape": [],
            "dtype": "python.float",
            "non_finite_count": 1,
        }
    if isinstance(value, dict):
        for key, item in value.items():
            found = _first_nonfinite(item, f"{name}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            found = _first_nonfinite(item, f"{name}[{index}]")
            if found is not None:
                return found
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            found = _first_nonfinite(getattr(value, field.name), f"{name}.{field.name}")
            if found is not None:
                return found
    return None


def _coordinate_checkpoint_health(
    failure: dict[str, Any] | None,
    *,
    context: str,
    rank: int,
    world_size: int,
    cpu_group,
    device: torch.device,
    epoch: int,
    optimizer_step: int,
) -> None:
    if world_size > 1:
        bad = torch.tensor([int(failure is not None)], dtype=torch.int32, device=device)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if not bool(bad.item()):
            return
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, failure, group=cpu_group)
        failures = [item for item in gathered if item is not None]
    elif failure is not None:
        failures = [failure]
    else:
        return
    for item in failures:
        print(
            json.dumps(
                {
                    "event": "checkpoint_numerical_failure",
                    "rank": item["rank"],
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    **item,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    detail = "; ".join(
        f"rank={item['rank']} tensor={item['tensor_name']} shape={item['shape']} "
        f"dtype={item['dtype']} non_finite_count={item['non_finite_count']}"
        for item in failures
    )
    raise FloatingPointError(f"Refusing {context} because non-finite state was found: {detail}")


def _validate_resume_candidate(checkpoint, *, model, optimizers, rank: int) -> None:
    """Validate all state that will be loaded before mutating live objects."""

    saved_state = checkpoint["model"]
    current_state = model.state_dict()
    missing = sorted(set(current_state) - set(saved_state))
    unexpected = sorted(set(saved_state) - set(current_state))
    if missing or unexpected:
        raise ValueError(
            f"Resume model state keys differ; missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name, expected in current_state.items():
        value = saved_state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Resume model state {name} is not a tensor")
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(
                f"Resume model state shape mismatch for {name}: "
                f"checkpoint={tuple(value.shape)}, target={tuple(expected.shape)}"
            )
        if value.dtype != expected.dtype:
            raise ValueError(
                f"Resume model state dtype mismatch for {name}: "
                f"checkpoint={value.dtype}, target={expected.dtype}"
            )
    runtime = checkpoint["rank_runtime"][rank]
    carry = runtime.get("carry")
    carried_ids = getattr(carry, "current_data", {}).get("puzzle_identifiers")
    puzzle_ids = (
        sorted({int(value) for value in carried_ids.detach().reshape(-1).cpu().tolist()})
        if isinstance(carried_ids, torch.Tensor)
        else []
    )
    failure_context = {
        "rank": rank,
        "epoch": int(checkpoint["completed_epochs"]),
        "optimizer_step": int(checkpoint["optimizer_step"]),
        "puzzle_identifiers": puzzle_ids,
    }

    def report_rejected_state(kind: str, bad: dict[str, Any]) -> None:
        record = {
            "event": "resume_checkpoint_rejected",
            **failure_context,
            "failure_kind": kind,
            **bad,
        }
        print(json.dumps(record, sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)

    bad_model = _first_nonfinite(saved_state, "model")
    if bad_model is not None:
        report_rejected_state("non_finite_model_state", bad_model)
        raise ValueError(f"Resume checkpoint has non-finite model state: {bad_model}")

    for optimizer_index, (optimizer, saved_optimizer) in enumerate(
        zip(optimizers, checkpoint["optimizers"])
    ):
        current_optimizer = optimizer.state_dict()
        saved_groups = saved_optimizer.get("param_groups")
        current_groups = current_optimizer.get("param_groups")
        if not isinstance(saved_groups, list) or len(saved_groups) != len(current_groups):
            raise ValueError(f"Resume optimizer {optimizer_index} has incompatible parameter groups")
        parameter_by_id: dict[Any, torch.Tensor] = {}
        for group_index, (saved_group, current_group, live_group) in enumerate(
            zip(saved_groups, current_groups, optimizer.param_groups)
        ):
            if not isinstance(saved_group, dict):
                raise ValueError(
                    f"Resume optimizer {optimizer_index} group {group_index} is malformed"
                )
            saved_ids = saved_group.get("params")
            current_ids = current_group.get("params")
            live_parameters = live_group["params"]
            if (
                not isinstance(saved_ids, list)
                or len(saved_ids) != len(current_ids)
                or len(saved_ids) != len(live_parameters)
            ):
                raise ValueError(
                    f"Resume optimizer {optimizer_index} group {group_index} has "
                    "incompatible parameter count"
                )
            parameter_by_id.update(zip(saved_ids, live_parameters))
        saved_state = saved_optimizer.get("state")
        if not isinstance(saved_state, dict):
            raise ValueError(f"Resume optimizer {optimizer_index} state is malformed")
        unknown_ids = set(saved_state) - set(parameter_by_id)
        if unknown_ids:
            raise ValueError(
                f"Resume optimizer {optimizer_index} has state for unknown parameter IDs: "
                f"{sorted(map(str, unknown_ids))[:8]}"
            )
        for parameter_id, state in saved_state.items():
            if not isinstance(state, dict):
                raise ValueError(
                    f"Resume optimizer {optimizer_index} parameter state is malformed"
                )
            parameter = parameter_by_id[parameter_id]
            for state_name, value in state.items():
                if isinstance(value, torch.Tensor):
                    if state_name != "step" and tuple(value.shape) != tuple(parameter.shape):
                        raise ValueError(
                            f"Resume optimizer state shape mismatch at optimizer[{optimizer_index}]"
                            f".state[{parameter_id}].{state_name}: checkpoint={tuple(value.shape)}, "
                            f"parameter={tuple(parameter.shape)}"
                        )
                elif not isinstance(value, (int, float, bool)):
                    raise ValueError(
                        f"Resume optimizer state optimizer[{optimizer_index}]"
                        f".state[{parameter_id}].{state_name} has unsupported type "
                        f"{type(value).__name__}"
                    )
        bad_optimizer = _first_nonfinite(saved_optimizer, f"optimizer[{optimizer_index}]")
        if bad_optimizer is not None:
            report_rejected_state("non_finite_optimizer_state", bad_optimizer)
            raise ValueError(f"Resume checkpoint has non-finite optimizer state: {bad_optimizer}")

    bad_carry = _first_nonfinite(carry, f"rank_runtime[{rank}].carry")
    if bad_carry is not None:
        report_rejected_state("non_finite_recurrent_carry", bad_carry)
        raise ValueError(f"Resume checkpoint has non-finite recurrent carry: {bad_carry}")


def _point_best_at_latest(latest: Path, best: Path) -> None:
    temporary = best.with_suffix(best.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(latest, temporary)
    except OSError:
        shutil.copy2(latest, temporary)
    os.replace(temporary, best)


def save_checkpoint(
    *,
    run_dir: Path,
    model,
    optimizers,
    config: dict[str, Any],
    epoch: int,
    optimizer_step: int,
    best_metrics: dict[str, Any],
    train_dataset,
    carry,
    rank: int,
    world_size: int,
    cpu_group,
    timing_summary: dict[str, float | int],
    is_best: bool,
) -> float:
    local_runtime = {
        "rng": capture_rng_state(),
        "carry": tree_map(carry, lambda tensor: tensor.detach().cpu()),
        "sampler_iters": int(train_dataset._iters),
    }
    local_failure = _first_nonfinite(
        {
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "carry": local_runtime["carry"],
        },
        "checkpoint_candidate",
    )
    if local_failure is not None:
        local_failure["rank"] = rank
        carry_ids = getattr(local_runtime["carry"], "current_data", {}).get(
            "puzzle_identifiers"
        )
        local_failure["puzzle_identifiers"] = (
            sorted({int(value) for value in carry_ids.detach().reshape(-1).cpu().tolist()})
            if isinstance(carry_ids, torch.Tensor)
            else []
        )
    model_device = next(model.parameters()).device
    _coordinate_checkpoint_health(
        local_failure,
        context=f"checkpoint save at epoch={epoch}, optimizer_step={optimizer_step}",
        rank=rank,
        world_size=world_size,
        cpu_group=cpu_group,
        device=model_device,
        epoch=epoch,
        optimizer_step=optimizer_step,
    )
    if world_size > 1:
        gathered_runtime = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_runtime, gathered_runtime, dst=0, group=cpu_group)
    else:
        gathered_runtime = [local_runtime]

    elapsed = 0.0
    if rank == 0:
        payload = {
            "format_version": 1,
            "schema_version": 2,
            "checkpoint_kind": "arc_agi_trm_training",
            "architecture": config["operator"],
            "architecture_hyperparameters": dict(config["model"]),
            "vocabulary": {
                "size": config["model"].get("vocab_size"),
                "output_size": config["model"].get("vocab_size"),
            },
            "source_revision": source_revision(),
            "completed_epochs": epoch,
            "optimizer_step": optimizer_step,
            "global_step": optimizer_step,
            "world_size": world_size,
            "config_fingerprint": checkpoint_fingerprint(config),
            "config": config,
            "model": tree_map(model.state_dict(), lambda tensor: tensor.detach().cpu()),
            "optimizers": [
                tree_map(optimizer.state_dict(), lambda tensor: tensor.detach().cpu())
                for optimizer in optimizers
            ],
            "scheduler": {
                "kind": "step_derived_warmup_cosine",
                "step": optimizer_step,
                "warmup_steps": int(config["optimization"]["lr_warmup_steps"]),
                "total_steps": max(1, int(config["estimated_total_optimizer_steps"])),
                "min_lr_ratio": float(config["optimization"]["lr_min_ratio"]),
            },
            # The training loop currently uses bfloat16 autocast without a GradScaler.
            "grad_scaler": None,
            "best_validation": best_metrics,
            "timing_summary": timing_summary,
            "rank_runtime": gathered_runtime,
        }
        elapsed = _atomic_torch_save(payload, run_dir / "latest.pt")
        if is_best:
            _point_best_at_latest(run_dir / "latest.pt", run_dir / "best.pt")
    if world_size > 1:
        dist.barrier()
    return elapsed


_save_checkpoint = save_checkpoint


def resume_training(
    *,
    path: Path,
    model,
    optimizers,
    config: dict[str, Any],
    train_dataset,
    rank: int,
    world_size: int,
    device: torch.device,
):
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = read_training_checkpoint(path, map_location="cpu", trusted=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    schema_version = int(checkpoint.get("schema_version", 1))
    if schema_version not in {1, 2}:
        raise ValueError(f"Unsupported training checkpoint schema {schema_version} in {path}")
    required_fields = (
        "completed_epochs", "optimizer_step", "config", "model", "optimizers",
        "scheduler", "rank_runtime", "best_validation", "world_size",
        "config_fingerprint",
    )
    missing_fields = [key for key in required_fields if key not in checkpoint]
    if missing_fields:
        raise ValueError(
            "Cannot resume exactly; training checkpoint is missing required state: "
            + ", ".join(missing_fields)
        )
    if schema_version >= 2 and checkpoint.get("architecture") != config.get("operator"):
        raise ValueError(
            f"Checkpoint architecture {checkpoint.get('architecture')!r} does not match "
            f"configured architecture {config.get('operator')!r}"
        )
    if int(checkpoint["world_size"]) != world_size:
        raise ValueError(
            f"Checkpoint was saved with world_size={checkpoint['world_size']}, "
            f"current world_size={world_size}; exact resume requires the same GPU count"
        )
    expected_fingerprint = checkpoint_fingerprint(config)
    if checkpoint.get("config_fingerprint") != expected_fingerprint:
        raise ValueError(
            "Resume configuration differs from the checkpoint in architecture, seed, "
            "data, optimizer settings, epoch budget, or run policy"
        )
    if int(checkpoint.get("scheduler", {}).get("step", -1)) != int(checkpoint["optimizer_step"]):
        raise ValueError("Checkpoint optimizer step and scheduler step are inconsistent")
    expected_scheduler = {
        "warmup_steps": int(config["optimization"]["lr_warmup_steps"]),
        "total_steps": max(1, int(config["estimated_total_optimizer_steps"])),
        "min_lr_ratio": float(config["optimization"]["lr_min_ratio"]),
    }
    saved_scheduler = checkpoint.get("scheduler", {})
    if schema_version >= 2 and saved_scheduler.get("kind") != "step_derived_warmup_cosine":
        raise ValueError("Checkpoint uses an unsupported scheduler state format")
    if any(saved_scheduler.get(key) != value for key, value in expected_scheduler.items()):
        raise ValueError(
            "Checkpoint scheduler state does not match the configured warmup, total "
            "step horizon, or minimum learning-rate ratio"
        )
    if schema_version >= 2 and int(checkpoint.get("global_step", -1)) != int(checkpoint["optimizer_step"]):
        raise ValueError("Checkpoint global_step and optimizer_step are inconsistent")
    if checkpoint.get("grad_scaler") is not None:
        raise ValueError("Checkpoint contains GradScaler state but this run has no GradScaler configured")
    if not isinstance(checkpoint["model"], dict) or not checkpoint["model"]:
        raise ValueError("Checkpoint has no model state; exact resume is unavailable")
    if len(checkpoint["optimizers"]) != len(optimizers):
        raise ValueError("Checkpoint optimizer count does not match the configured optimizer setup")
    if any(not isinstance(state, dict) for state in checkpoint["optimizers"]):
        raise ValueError("Checkpoint contains a malformed optimizer state")
    if len(checkpoint["rank_runtime"]) != world_size:
        raise ValueError("Checkpoint does not contain RNG/carry state for every rank")
    if rank < 0 or rank >= len(checkpoint["rank_runtime"]):
        raise ValueError(f"Checkpoint has no runtime state for rank {rank}")
    runtime = checkpoint["rank_runtime"][rank]
    if not isinstance(runtime, dict) or not {"rng", "carry", "sampler_iters"}.issubset(runtime):
        raise ValueError(f"Checkpoint rank {rank} is missing RNG, carry, or sampler state")
    if not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(runtime["rng"]):
        raise ValueError(f"Checkpoint rank {rank} is missing one or more RNG states")
    if runtime["rng"]["torch_cuda"] and not torch.cuda.is_available():
        raise ValueError(
            "Checkpoint has CUDA RNG state but CUDA is unavailable; exact resume cannot restore it"
        )
    if int(checkpoint["completed_epochs"]) > 0 and runtime["carry"] is None:
        raise ValueError("Checkpoint is missing the TRM carry required to continue the next epoch")
    _validate_resume_candidate(
        checkpoint,
        model=model,
        optimizers=optimizers,
        rank=rank,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
        optimizer.load_state_dict(state)
    restore_rng_state(runtime["rng"])
    train_dataset._iters = int(runtime["sampler_iters"])
    carry = tree_map(runtime["carry"], lambda tensor: tensor.to(device))
    return (
        int(checkpoint["completed_epochs"]),
        int(checkpoint["optimizer_step"]),
        carry,
        checkpoint.get("best_validation", {}),
        checkpoint.get("timing_summary", {"epoch_duration_sum": 0.0, "epoch_duration_count": 0}),
    )


_load_resume = resume_training


def _reduce_epoch_result(
    result,
    device: torch.device,
    world_size: int,
    *,
    diagnostics: NumericsDiagnostics | None = None,
    epoch: int = 0,
):
    keys = sorted(result.metric_sums_local)
    values = [result.loss_sum_local] + [result.metric_sums_local[key] for key in keys]
    values.append(float(result.local_rows_forwarded))
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if diagnostics is not None:
        diagnostics.set_context(epoch=epoch, optimizer_step=result.optimizer_step)
        diagnostics.check_tensor("epoch.reduced_sums", tensor, stage="epoch_reduction")
        diagnostics.synchronize("epoch_reduction", device=device)
    reduced = tensor.cpu().tolist()
    return reduced[0], {key: reduced[i + 1] for i, key in enumerate(keys)}, reduced[-1]


def _load_evaluation_artifacts(dataset_dir: Path):
    with (dataset_dir / "identifiers.json").open("r", encoding="utf-8") as stream:
        identifiers = json.load(stream)
    with (dataset_dir / "test_puzzles.json").open("r", encoding="utf-8") as stream:
        test_puzzles = json.load(stream)
    return identifiers, test_puzzles


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _runtime_environment(device: torch.device, world_size: int) -> dict[str, Any]:
    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        if hasattr(torch._C, "_GLIBCXX_USE_CXX11_ABI")
        else None,
        "mamba_ssm": package_version("mamba-ssm"),
        "causal_conv1d": package_version("causal-conv1d"),
        "world_size": world_size,
        "current_device": torch.cuda.get_device_name(device),
        "visible_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def run(args: argparse.Namespace) -> int:
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are separate modes; choose only one")
    if args.stop_after_step is not None and args.stop_after_step < 1:
        raise ValueError("--stop-after-step must be positive")
    if args.smoke_test_steps is not None and args.smoke_test_steps < 1:
        raise ValueError("--smoke-test-steps must be positive")
    if args.smoke_test_steps is not None and any(
        (args.resume, args.init_from, args.stop_after_step, args.diagnose_numerics)
    ):
        raise ValueError(
            "--smoke-test-steps uses a fresh, bounded run; do not combine it with "
            "--resume, --init-from, --stop-after-step, or --diagnose-numerics"
        )
    if args.smoke_test_steps is not None and args.run_dir is None:
        raise ValueError("--smoke-test-steps requires an explicit fresh --run-dir")
    if args.diagnose_numerics and args.stop_after_step is None:
        raise ValueError(
            "--diagnose-numerics requires --stop-after-step so diagnostic runs remain bounded"
        )
    rank, world_size, _local_rank, cpu_group, device = _init_distributed()
    progress = None
    progress_status = "incomplete"
    optimizer_step = 0
    config = load_experiment_config(args.config)
    config["source_config"] = str(Path(args.config).resolve())
    run_dir = _resolve_from_root(args.run_dir or config["run_dir"])
    config["run_dir"] = str(run_dir)
    dataset_dir = _resolve_from_root(args.dataset)
    try:
        manifest = load_dataset_manifest(dataset_dir)
        config["dataset_manifest_fingerprint"] = manifest_fingerprint(manifest)
        check_dataset_config(manifest, config, allow_small=args.allow_small_dataset)
        batch_plan = _validate_config(config, world_size)

        if args.smoke_test_steps is not None and run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Smoke-test output directory must be fresh and empty: {run_dir}"
            )
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()
        if args.diagnose_numerics and any(
            (run_dir / name).exists() for name in ("latest.pt", "best.pt")
        ):
            raise FileExistsError(
                f"Numerics diagnostic output directory already contains a checkpoint: {run_dir}. "
                "Choose a fresh --run-dir so the source checkpoint remains untouched."
            )

        global_batch_size = int(config["optimization"]["global_batch_size"])
        train_loader, train_dataset, validation_loader, validation_dataset = create_loaders(
            dataset_dir,
            seed=int(config["seed"]),
            global_batch_size=global_batch_size,
            rank=rank,
            world_size=world_size,
        )
        optimizer_updates_by_epoch = estimate_train_optimizer_updates(
            train_dataset,
            epochs=int(config["epochs"]),
            accumulation_steps=int(config["optimization"]["gradient_accumulation_steps"]),
        )
        optimizer_steps_by_epoch = [0]
        for updates in optimizer_updates_by_epoch:
            optimizer_steps_by_epoch.append(optimizer_steps_by_epoch[-1] + updates)
        if len(optimizer_updates_by_epoch) != int(config["epochs"]):
            raise RuntimeError(
                "Progress planner did not return one sampler count per configured epoch"
            )
        empty_epochs = [
            epoch
            for epoch, updates in enumerate(optimizer_updates_by_epoch, start=1)
            if updates <= 0
        ]
        if empty_epochs:
            raise RuntimeError(
                "The upstream sampler plan contains zero optimizer updates for epoch(s) "
                f"{empty_epochs}; refusing to start a 0/0 progress bar or allocate the model. "
                "Check the training dataset, global_batch_size, world_size, and "
                "gradient_accumulation_steps."
            )
        planned_total_steps = optimizer_steps_by_epoch[-1]
        config["estimated_total_optimizer_steps"] = planned_total_steps
        config["model"].update(
            batch_size=batch_plan["per_device_batch_size"],
            seq_len=int(train_dataset.metadata.seq_len),
            vocab_size=int(train_dataset.metadata.vocab_size),
            num_puzzle_identifiers=int(train_dataset.metadata.num_puzzle_identifiers),
        )
        if int(config["model"]["batch_size"]) != batch_plan["per_device_batch_size"]:
            raise AssertionError("Model batch_size diverged from the resolved per-device batch")
        batch_plan["validation_per_device_batch_size"] = validation_dataset.local_batch_size
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "resolved_training_plan",
                        "operator": config["operator"],
                        "world_size": world_size,
                        "per_device_batch_size": batch_plan["per_device_batch_size"],
                        "global_batch_size": batch_plan["microbatch_global_size"],
                        "gradient_accumulation_steps": batch_plan[
                            "gradient_accumulation_steps"
                        ],
                        "effective_global_batch_size": batch_plan[
                            "effective_global_batch_size"
                        ],
                        "optimizer_updates_per_epoch": optimizer_updates_by_epoch,
                        "total_optimizer_steps": planned_total_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        # All workers build identical initial weights; rank-specific RNG streams
        # are installed after construction and restored from rank-local checkpoints.
        seed_everything(int(config["seed"]))
        loss_model = create_experiment_model(
            config,
            train_dataset.metadata,
            per_device_batch_size=batch_plan["per_device_batch_size"],
            device=device,
        )
        optimizers, optimizer_base_lrs = create_optimizers(
            loss_model, config, world_size=world_size
        )
        if args.resume:
            resume_path = Path(args.resume)
            if str(args.resume).lower() == "latest":
                resume_path = run_dir / "latest.pt"
            elif resume_path.is_dir():
                resume_path = resume_path / "latest.pt"
            completed_epochs, optimizer_step, carry, best_metrics, timing_summary = resume_training(
                path=resume_path.resolve(),
                model=loss_model,
                optimizers=optimizers,
                config=config,
                train_dataset=train_dataset,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "resume_checkpoint_validated",
                            "path": str(resume_path.resolve()),
                            "completed_epochs": completed_epochs,
                            "optimizer_step": optimizer_step,
                            "world_size": world_size,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        else:
            completed_epochs, optimizer_step, carry, best_metrics = 0, 0, None, {}
            timing_summary = {"epoch_duration_sum": 0.0, "epoch_duration_count": 0}
            if args.init_from:
                report = load_pretrained(
                    loss_model,
                    _resolve_from_root(args.init_from),
                    device=device,
                    strict=True,
                    expected_config=config,
                )
                if rank == 0:
                    print(
                        "Initialized model weights from pretrained artifact; optimizer and "
                        f"scheduler start fresh: {report['architecture']}",
                        flush=True,
                    )
            seed_training_rank(int(config["seed"]), rank)

        progress_final_epoch = int(config["epochs"])
        if args.stop_after_step is not None:
            first_epoch_to_run = min(int(config["epochs"]), completed_epochs + 1)
            if optimizer_step >= args.stop_after_step:
                progress_final_epoch = first_epoch_to_run
            else:
                for candidate_epoch in range(first_epoch_to_run, int(config["epochs"]) + 1):
                    if optimizer_steps_by_epoch[candidate_epoch] >= args.stop_after_step:
                        progress_final_epoch = candidate_epoch
                        break
        progress_total_steps = max(optimizer_step, optimizer_steps_by_epoch[progress_final_epoch])
        if progress_total_steps <= 0:
            raise RuntimeError(
                "Resolved progress total is zero; refusing to start with a 0/0 optimizer-step bar"
            )

        identifiers, test_puzzles = _load_evaluation_artifacts(dataset_dir)
        diagnostics = None
        if args.diagnose_numerics:
            diagnostics = NumericsDiagnostics(
                rank=rank,
                world_size=world_size,
                cpu_group=cpu_group,
                log_path=run_dir / f"numerics_rank{rank}.jsonl",
                puzzle_names=identifiers,
            )
            _install_numerics_hooks(loss_model, diagnostics)
            if rank == 0:
                print(
                    f"Opt-in numerical diagnostics enabled; per-rank logs: "
                    f"{run_dir / 'numerics_rank*.jsonl'}",
                    flush=True,
                )

        if rank == 0:
            (run_dir / "config_snapshot.json").write_text(
                json.dumps(
                    {
                        "experiment": config,
                        "dataset_manifest": manifest,
                        "runtime_environment": _runtime_environment(device, world_size),
                        "batch_sizes": batch_plan,
                        "planned_optimizer_steps": planned_total_steps,
                        "progress_optimizer_updates_per_epoch": optimizer_updates_by_epoch,
                        "progress_total_optimizer_steps": progress_total_steps,
                        "model_parameter_count": sum(
                            parameter.numel() for parameter in loss_model.parameters()
                        ),
                        "trainable_mamba_parameter_count": sum(
                            parameter.numel()
                            for layer in loss_model.model.inner.L_level.layers
                            for mamba in _mamba_modules(layer)
                            for parameter in mamba.parameters()
                            if parameter.requires_grad
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            with (run_dir / "run.log").open("a", encoding="utf-8") as stream:
                stream.write(
                    f"Starting {config['name']} at completed_epoch={completed_epochs}, "
                    f"world_size={world_size}, device={device}, architecture={config['operator']}, "
                    f"model={json.dumps(config['model'], sort_keys=True)}\n"
                )
                stream.flush()

        if not args.resume and args.smoke_test_steps is None:
            save_checkpoint(
                run_dir=run_dir,
                model=loss_model,
                optimizers=optimizers,
                config=config,
                epoch=0,
                optimizer_step=0,
                best_metrics=best_metrics,
                train_dataset=train_dataset,
                carry=None,
                rank=rank,
                world_size=world_size,
                cpu_group=cpu_group,
                timing_summary=timing_summary,
                is_best=False,
            )

        if rank == 0:
            progress = TrainingProgress(
                total_steps=(
                    args.smoke_test_steps
                    if args.smoke_test_steps is not None
                    else progress_total_steps
                ),
                initial_step=optimizer_step,
                initial_epoch=min(int(config["epochs"]), completed_epochs + 1),
                total_epochs=int(config["epochs"]),
            )

        metrics_path = run_dir / "metrics.jsonl"
        if args.smoke_test_steps is not None:
            smoke_memory_path = run_dir / "smoke_memory.jsonl"
            accumulation_steps = int(config["optimization"]["gradient_accumulation_steps"])
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            last_memory_record: dict[str, int] | None = None

            def record_smoke_step(update: dict[str, Any]) -> None:
                nonlocal last_memory_record
                if progress is not None:
                    progress.optimizer_step(update)
                torch.cuda.synchronize(device)
                peak_memory = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(device),
                        torch.cuda.max_memory_reserved(device),
                    ],
                    dtype=torch.int64,
                    device="cpu",
                )
                if world_size > 1:
                    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX, group=cpu_group)
                values = peak_memory.tolist()
                last_memory_record = {
                    "peak_allocated_bytes_max_rank": int(values[0]),
                    "peak_reserved_bytes_max_rank": int(values[1]),
                }
                if rank == 0:
                    memory_record = {
                        "event": "smoke_optimizer_step",
                        "optimizer_step": int(update["optimizer_step"]),
                        "training_loss": float(update["training_loss"]),
                        "learning_rate": float(update["learning_rate"]),
                        "examples": int(update["examples"]),
                        "world_size": world_size,
                        **last_memory_record,
                    }
                    _append_jsonl(smoke_memory_path, memory_record)
                    if progress is not None:
                        progress.write(json.dumps(memory_record, sort_keys=True))
                torch.cuda.reset_peak_memory_stats(device)

            try:
                smoke_result = train_one_epoch(
                    model=loss_model,
                    loader=islice(
                        train_loader,
                        int(args.smoke_test_steps) * accumulation_steps,
                    ),
                    optimizers=optimizers,
                    optimizer_base_lrs=optimizer_base_lrs,
                    device=device,
                    carry=None,
                    optimizer_step=optimizer_step,
                    total_steps=planned_total_steps,
                    warmup_steps=int(config["optimization"]["lr_warmup_steps"]),
                    min_lr_ratio=float(config["optimization"]["lr_min_ratio"]),
                    global_batch_size=global_batch_size,
                    accumulation_steps=accumulation_steps,
                    gradient_clip_norm=config["optimization"].get("gradient_clip_norm"),
                    rank=rank,
                    world_size=world_size,
                    epoch=completed_epochs + 1,
                    vocab_size=int(config["model"]["vocab_size"]),
                    num_puzzle_identifiers=int(config["model"]["num_puzzle_identifiers"]),
                    on_optimizer_step=record_smoke_step,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA out of memory in {config['operator']} smoke test on rank={rank}, "
                    f"per_device_batch_size={batch_plan['per_device_batch_size']}, "
                    f"global_batch_size={global_batch_size}, "
                    f"gradient_accumulation_steps={accumulation_steps}. "
                    "The torchrun worker agent will stop the peer ranks; retry with the "
                    "smaller explicit per-device-batch preset."
                ) from exc
            completed_smoke_steps = smoke_result.optimizer_step - optimizer_step
            if completed_smoke_steps != int(args.smoke_test_steps):
                raise RuntimeError(
                    f"Smoke test requested {args.smoke_test_steps} optimizer steps but "
                    f"the loader completed {completed_smoke_steps}; check the dataset and "
                    "accumulation configuration."
                )
            if rank == 0:
                summary = {
                    "event": "smoke_test_completed",
                    "optimizer_steps": completed_smoke_steps,
                    "examples_sampled": smoke_result.examples_sampled,
                    "world_size": world_size,
                    "per_device_batch_size": batch_plan["per_device_batch_size"],
                    "global_batch_size": global_batch_size,
                    "gradient_accumulation_steps": accumulation_steps,
                    "memory_log": str(smoke_memory_path),
                    **(last_memory_record or {}),
                }
                _append_jsonl(metrics_path, summary)
                with (run_dir / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(summary, sort_keys=True) + "\n")
                    stream.flush()
                if progress is not None:
                    progress.write(json.dumps(summary, sort_keys=True))
                progress_status = "smoke-complete"
            return 0

        for epoch in range(completed_epochs + 1, int(config["epochs"]) + 1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            epoch_started = time.perf_counter()
            try:
                epoch_result = train_one_epoch(
                    model=loss_model,
                    loader=train_loader,
                    optimizers=optimizers,
                    optimizer_base_lrs=optimizer_base_lrs,
                    device=device,
                    carry=carry,
                    optimizer_step=optimizer_step,
                    total_steps=planned_total_steps,
                    warmup_steps=int(config["optimization"]["lr_warmup_steps"]),
                    min_lr_ratio=float(config["optimization"]["lr_min_ratio"]),
                    global_batch_size=global_batch_size,
                    accumulation_steps=int(config["optimization"]["gradient_accumulation_steps"]),
                    gradient_clip_norm=config["optimization"].get("gradient_clip_norm"),
                    rank=rank,
                    world_size=world_size,
                    epoch=epoch,
                    diagnostics=diagnostics,
                    vocab_size=int(config["model"]["vocab_size"]),
                    num_puzzle_identifiers=int(config["model"]["num_puzzle_identifiers"]),
                    on_optimizer_step=(
                        (lambda update: progress.optimizer_step(update) if progress else None)
                    ),
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA out of memory in {config['operator']} training on rank={rank} at "
                    f"epoch={epoch}, per_device_batch_size={batch_plan['per_device_batch_size']}, "
                    f"global_batch_size={global_batch_size}, "
                    f"gradient_accumulation_steps={batch_plan['gradient_accumulation_steps']}. "
                    "The torchrun worker agent will stop the peer ranks. Reduce the "
                    "configured per-device batch size in an explicit run preset and retry."
                ) from exc
            if epoch_result.examples_sampled <= 0:
                raise RuntimeError(
                    f"Sampler emitted no complete batches in epoch {epoch}; the epoch was not completed"
                )
            actual_epoch_updates = epoch_result.optimizer_step - optimizer_step
            predicted_epoch_updates = optimizer_updates_by_epoch[epoch - 1]
            if actual_epoch_updates != predicted_epoch_updates and rank == 0:
                message = (
                    f"Progress estimate adjusted: sampler emitted {actual_epoch_updates} "
                    f"optimizer updates in epoch {epoch}; planned {predicted_epoch_updates}."
                )
                if progress:
                    progress.write(message)
                else:
                    print(message, flush=True)
            if actual_epoch_updates != predicted_epoch_updates:
                optimizer_updates_by_epoch[epoch - 1] = actual_epoch_updates
                optimizer_steps_by_epoch = [0]
                for updates in optimizer_updates_by_epoch:
                    optimizer_steps_by_epoch.append(optimizer_steps_by_epoch[-1] + updates)
                planned_total_steps = optimizer_steps_by_epoch[-1]
                config["estimated_total_optimizer_steps"] = planned_total_steps
            carry = epoch_result.carry
            optimizer_step = epoch_result.optimizer_step
            if args.stop_after_step is not None and optimizer_step >= args.stop_after_step:
                progress_total_steps = optimizer_step
            else:
                progress_total_steps = optimizer_step
                for future_epoch in range(epoch, int(config["epochs"])):
                    progress_total_steps += optimizer_updates_by_epoch[future_epoch]
                    if (
                        args.stop_after_step is not None
                        and progress_total_steps >= args.stop_after_step
                    ):
                        break
            if progress is not None:
                progress.set_total(progress_total_steps)
            loss_sum, reduced_metrics, rows_forwarded = _reduce_epoch_result(
                epoch_result,
                device,
                world_size,
                diagnostics=diagnostics,
                epoch=epoch,
            )
            training_seconds = time.perf_counter() - epoch_started
            train_loss = loss_sum / max(1, epoch_result.examples_sampled)
            train_count = max(1.0, reduced_metrics.get("count", 0.0))
            record: dict[str, Any] = {
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "epoch_seconds": None,
                "average_epoch_seconds": None,
                "training_seconds": training_seconds,
                "training_examples_sampled": epoch_result.examples_sampled,
                "training_examples_per_second": epoch_result.examples_sampled / max(training_seconds, 1e-9),
                "training_rows_forwarded": int(rows_forwarded),
                "training_loss": train_loss,
                "training_lm_loss": reduced_metrics.get("lm_loss", 0.0)
                / max(1, epoch_result.examples_sampled),
                "training_q_halt_loss": reduced_metrics.get("q_halt_loss", 0.0)
                / max(1, epoch_result.examples_sampled),
                "training_grid_accuracy": reduced_metrics.get("accuracy", 0.0) / train_count,
                "training_batch_exact_accuracy": reduced_metrics.get("exact_accuracy", 0.0)
                / train_count,
                "training_rows_halted": int(reduced_metrics.get("count", 0.0)),
                "learning_rate": epoch_result.learning_rates[-1]
                if epoch_result.learning_rates
                else None,
                "seed": config["seed"],
                "operator": config["operator"],
                "architecture": config["operator"],
                "model_hyperparameters": config["model"],
                "dataset_split": {
                    "validation_tasks": int(config["data"]["validation_tasks"]),
                    "validation_variants_per_task": int(config["data"]["validation_variants_per_task"]),
                },
                "augmentation": config["data"],
                "optimization": config["optimization"],
                "precision": config["model"].get("forward_dtype"),
                "checkpoint_paths": {
                    "latest": str(run_dir / "latest.pt"),
                    "best": str(run_dir / "best.pt"),
                    "best_pretrained": str(run_dir / "best_pretrained.pt"),
                    "final_pretrained": str(run_dir / "final_pretrained.pt"),
                },
                "world_size": world_size,
                "global_batch_size": batch_plan["microbatch_global_size"],
                "per_device_batch_size": batch_plan["per_device_batch_size"],
                "gradient_accumulation_steps": batch_plan["gradient_accumulation_steps"],
                "effective_global_batch_size": batch_plan["effective_global_batch_size"],
                "validation_loss": None,
                "validation_task_exact_match": None,
                "validation_tasks_evaluated": None,
                "validation_tasks_solved": None,
                "validation_inference_seconds": None,
                "checkpoint_path": None,
                "checkpoint_write_seconds": 0.0,
            }
            should_evaluate = (
                epoch % int(config["run"]["eval_every_epochs"]) == 0
                or epoch == int(config["epochs"])
            )
            validation_metrics = None
            improved = False
            if should_evaluate:
                validation_metrics = evaluate_validation(
                    loss_model=loss_model,
                    loader=validation_loader,
                    identifiers=identifiers,
                    test_puzzles=test_puzzles,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    cpu_group=cpu_group,
                )
                if rank == 0:
                    previous = best_metrics
                    improved = validation_improved(validation_metrics, previous)
                    if improved:
                        best_metrics = dict(validation_metrics)
                        best_metrics["epoch"] = epoch
                        best_metrics["optimizer_step"] = optimizer_step
                if world_size > 1:
                    decision = [
                        {"improved": improved, "best": best_metrics} if rank == 0 else None
                    ]
                    dist.broadcast_object_list(decision, src=0, group=cpu_group)
                    improved = bool(decision[0]["improved"])
                    best_metrics = decision[0]["best"]
                if rank == 0 and validation_metrics is not None:
                    record.update(validation_metrics)

            epoch_seconds = time.perf_counter() - epoch_started
            timing_summary = {
                "epoch_duration_sum": float(timing_summary["epoch_duration_sum"]) + epoch_seconds,
                "epoch_duration_count": int(timing_summary["epoch_duration_count"]) + 1,
            }
            record["epoch_seconds"] = epoch_seconds
            record["average_epoch_seconds"] = float(timing_summary["epoch_duration_sum"]) / max(
                1, int(timing_summary["epoch_duration_count"])
            )

            stop_after_step = bool(
                args.stop_after_step is not None
                and optimizer_step >= args.stop_after_step
            )
            if stop_after_step:
                record["diagnostic_stop_after_step"] = args.stop_after_step
            should_checkpoint = (
                epoch % int(config["run"]["checkpoint_every_epochs"]) == 0
                or epoch == int(config["epochs"])
                or improved
                or stop_after_step
            )
            if should_checkpoint:
                record["checkpoint_write_seconds"] = save_checkpoint(
                    run_dir=run_dir,
                    model=loss_model,
                    optimizers=optimizers,
                    config=config,
                    epoch=epoch,
                    optimizer_step=optimizer_step,
                    best_metrics=best_metrics,
                    train_dataset=train_dataset,
                    carry=carry,
                    rank=rank,
                    world_size=world_size,
                    cpu_group=cpu_group,
                    timing_summary=timing_summary,
                    is_best=improved,
                )
                record["checkpoint_path"] = str(run_dir / "latest.pt")
                if improved and rank == 0:
                    save_pretrained(
                        loss_model,
                        run_dir / "best_pretrained.pt",
                        config,
                        metadata={
                            "epoch": epoch,
                            "global_step": optimizer_step,
                            "best_validation": best_metrics,
                            "experiment_name": config.get("name"),
                        },
                    )
            record["peak_gpu_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
            )
            record["peak_gpu_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
            )
            if rank == 0:
                _append_jsonl(metrics_path, record)
                with (run_dir / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                if progress:
                    progress.epoch_completed(epoch, record)
                    progress.write(json.dumps(record, sort_keys=True, allow_nan=False))
                else:
                    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
            completed_epochs = epoch
            if stop_after_step:
                break

        if rank == 0:
            if args.stop_after_step is not None and optimizer_step >= args.stop_after_step:
                message = (
                    f"Stopped at epoch boundary after optimizer_step={optimizer_step}; "
                    f"diagnostic checkpoint={run_dir / 'latest.pt'}"
                )
                if progress:
                    progress.write(message)
                else:
                    print(message, flush=True)
                progress_status = "diagnostic-stop"
            else:
                save_pretrained(
                    loss_model,
                    run_dir / "final_pretrained.pt",
                    config,
                    metadata={
                        "completed_epochs": int(config["epochs"]),
                        "global_step": optimizer_step,
                        "best_validation": best_metrics,
                        "experiment_name": config.get("name"),
                    },
                )
                message = (
                    f"Completed {config['epochs']} epochs for {config['name']}; "
                    f"latest={run_dir / 'latest.pt'}, best={run_dir / 'best.pt'}"
                )
                if progress:
                    progress.write(message)
                else:
                    print(message, flush=True)
                progress_status = "complete"
        return 0
    finally:
        if progress is not None:
            progress.close(final_step=optimizer_step, status=progress_status)
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, help="Override the run preset output directory")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume from RUN_DIR/latest.pt, a checkpoint file, or a checkpoint directory",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        help="Initialize model weights from a model-only artifact; create fresh optimizer/scheduler state",
    )
    parser.add_argument(
        "--allow-small-dataset",
        action="store_true",
        help="Permit a non-1,000-task fixture for development only",
    )
    parser.add_argument(
        "--diagnose-numerics",
        action="store_true",
        help="Enable opt-in tensor, gradient, parameter, optimizer-state, and distributed finite checks",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Stop at the first completed epoch boundary at or after this optimizer step",
    )
    parser.add_argument(
        "--smoke-test-steps",
        type=int,
        help=(
            "Run this many optimizer updates from a fresh model, skip validation and "
            "checkpoints, and log peak memory for a bounded forward/backward test"
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        print(
            "Interrupted between checkpoint boundaries. Resume from the latest valid "
            "completed-epoch checkpoint; the incomplete epoch was not saved.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
