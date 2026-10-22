"""Train a project-owned TRM operator with the unchanged upstream TRM loop."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
import traceback
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
    evaluate_validation,
    load_dataset_manifest,
)


def _resolve_from_root(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _init_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment entry point requires CUDA. Mamba-2 operators also require "
            "the qualified mamba-ssm/causal-conv1d kernels. No CUDA device is available; "
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
    run_settings = config["run"]
    if int(run_settings.get("num_workers", 0)) != 0:
        raise ValueError(
            "Exact resume is configured with num_workers=0 so sampler RNG state can be saved"
        )
    if int(run_settings["eval_every_epochs"]) < 1 or int(run_settings["checkpoint_every_epochs"]) < 1:
        raise ValueError("Evaluation and checkpoint intervals must be positive")
    if int(run_settings["eval_attempts"]) != 1:
        raise ValueError("This preset uses one greedy top-1 validation attempt")
    return batch_sizes(
        int(config["optimization"]["global_batch_size"]),
        world_size,
        int(config["optimization"]["gradient_accumulation_steps"]),
    )


def _mamba_modules(layer):
    if hasattr(layer, "mamba"):
        return (layer.mamba,)
    if hasattr(layer, "mamba1") and hasattr(layer, "mamba2"):
        return (layer.mamba1, layer.mamba2)
    return ()


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


def _reduce_epoch_result(result, device: torch.device, world_size: int):
    keys = sorted(result.metric_sums_local)
    values = [result.loss_sum_local] + [result.metric_sums_local[key] for key in keys]
    values.append(float(result.local_rows_forwarded))
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
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
    rank, world_size, _local_rank, cpu_group, device = _init_distributed()
    config = load_experiment_config(args.config)
    config["source_config"] = str(Path(args.config).resolve())
    run_dir = _resolve_from_root(args.run_dir or config["run_dir"])
    dataset_dir = _resolve_from_root(args.dataset)
    try:
        manifest = load_dataset_manifest(dataset_dir)
        config["dataset_manifest_fingerprint"] = manifest_fingerprint(manifest)
        check_dataset_config(manifest, config, allow_small=args.allow_small_dataset)
        batch_plan = _validate_config(config, world_size)

        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()

        global_batch_size = int(config["optimization"]["global_batch_size"])
        train_loader, train_dataset, validation_loader, validation_dataset = create_loaders(
            dataset_dir,
            seed=int(config["seed"]),
            global_batch_size=global_batch_size,
            rank=rank,
            world_size=world_size,
        )
        config["model"].update(
            batch_size=batch_plan["per_device_batch_size"],
            seq_len=int(train_dataset.metadata.seq_len),
            vocab_size=int(train_dataset.metadata.vocab_size),
            num_puzzle_identifiers=int(train_dataset.metadata.num_puzzle_identifiers),
        )
        batch_plan["validation_per_device_batch_size"] = validation_dataset.local_batch_size
        examples_per_epoch_estimate = (
            train_dataset.metadata.total_groups * train_dataset.metadata.mean_puzzle_examples
        )
        estimated_total_steps = max(
            1,
            int(
                int(config["epochs"])
                * examples_per_epoch_estimate
                / (
                    global_batch_size
                    * int(config["optimization"]["gradient_accumulation_steps"])
                )
            ),
        )
        config["estimated_total_optimizer_steps"] = estimated_total_steps

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

        if rank == 0:
            (run_dir / "config_snapshot.json").write_text(
                json.dumps(
                    {
                        "experiment": config,
                        "dataset_manifest": manifest,
                        "runtime_environment": _runtime_environment(device, world_size),
                        "batch_sizes": batch_plan,
                        "estimated_optimizer_steps": estimated_total_steps,
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

        if not args.resume:
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

        identifiers, test_puzzles = _load_evaluation_artifacts(dataset_dir)
        metrics_path = run_dir / "metrics.jsonl"
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
                    total_steps=estimated_total_steps,
                    warmup_steps=int(config["optimization"]["lr_warmup_steps"]),
                    min_lr_ratio=float(config["optimization"]["lr_min_ratio"]),
                    global_batch_size=global_batch_size,
                    accumulation_steps=int(config["optimization"]["gradient_accumulation_steps"]),
                    gradient_clip_norm=config["optimization"].get("gradient_clip_norm"),
                    rank=rank,
                    world_size=world_size,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA out of memory in {config['operator']} training at "
                    f"epoch={epoch}, per_device_batch_size={batch_plan['per_device_batch_size']}, "
                    f"global_batch_size={global_batch_size}, "
                    f"gradient_accumulation_steps={batch_plan['gradient_accumulation_steps']}. "
                    "Reduce the configured per-device batch size and rebuild the run config."
                ) from exc
            if epoch_result.examples_sampled <= 0:
                raise RuntimeError(
                    f"Sampler emitted no complete batches in epoch {epoch}; the epoch was not completed"
                )
            carry = epoch_result.carry
            optimizer_step = epoch_result.optimizer_step
            loss_sum, reduced_metrics, rows_forwarded = _reduce_epoch_result(
                epoch_result, device, world_size
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

            should_checkpoint = (
                epoch % int(config["run"]["checkpoint_every_epochs"]) == 0
                or epoch == int(config["epochs"])
                or improved
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
                print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
            completed_epochs = epoch

        if rank == 0:
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
            print(
                f"Completed {config['epochs']} epochs for {config['name']}; "
                f"latest={run_dir / 'latest.pt'}, best={run_dir / 'best.pt'}",
                flush=True,
            )
        return 0
    finally:
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
