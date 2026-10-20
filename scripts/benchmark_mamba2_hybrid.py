"""Measure a full ARC-shaped Mamba-2 hybrid training/evaluation step on CUDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from arc_agi_trm.experiment import (  # noqa: E402
    batch_sizes,
    load_experiment_config,
    seed_everything,
    seed_training_rank,
    train_one_epoch,
    tree_map,
)
from arc_agi_trm.runner import (  # noqa: E402
    check_dataset_config,
    create_hybrid_model,
    create_loaders,
    create_optimizers,
    evaluate_validation,
    load_dataset_manifest,
)
from train_mamba2_hybrid import _atomic_torch_save, _init_distributed, _validate_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("warmup must be non-negative and steps must be positive")

    rank, world_size, _local_rank, cpu_group, device = _init_distributed()
    try:
        config = load_experiment_config(args.config)
        batch_plan = _validate_config(config, world_size)
        manifest = load_dataset_manifest(args.dataset)
        check_dataset_config(manifest, config, allow_small=args.allow_small_dataset)
        batch_size = int(config["optimization"]["global_batch_size"])
        train_loader, _train_dataset, validation_loader, validation_dataset = create_loaders(
            args.dataset,
            seed=int(config["seed"]),
            global_batch_size=batch_size,
            rank=rank,
            world_size=world_size,
        )
        batch_plan["validation_per_device_batch_size"] = validation_dataset.local_batch_size
        seed_everything(int(config["seed"]))
        loss_model = create_hybrid_model(
            config,
            validation_dataset.metadata,
            per_device_batch_size=batch_plan["per_device_batch_size"],
            device=device,
        )
        optimizers, optimizer_base_lrs = create_optimizers(
            loss_model, config, world_size=world_size
        )
        seed_training_rank(int(config["seed"]), rank)
        total_steps = max(
            1,
            int(
                int(config["epochs"])
                * _train_dataset.metadata.total_groups
                * _train_dataset.metadata.mean_puzzle_examples
                / (
                    batch_size
                    * int(config["optimization"]["gradient_accumulation_steps"])
                )
            ),
        )
        batch_iterator = iter(train_loader)
        carry = None
        optimizer_step = 0
        warmup_start = time.perf_counter()
        for _ in range(args.warmup):
            batch = next(batch_iterator)
            result = train_one_epoch(
                model=loss_model,
                loader=[batch],
                optimizers=optimizers,
                optimizer_base_lrs=optimizer_base_lrs,
                device=device,
                carry=carry,
                optimizer_step=optimizer_step,
                total_steps=total_steps,
                warmup_steps=int(config["optimization"]["lr_warmup_steps"]),
                min_lr_ratio=float(config["optimization"]["lr_min_ratio"]),
                global_batch_size=batch_size,
                accumulation_steps=1,
                gradient_clip_norm=config["optimization"].get("gradient_clip_norm"),
                rank=rank,
                world_size=world_size,
            )
            carry, optimizer_step = result.carry, result.optimizer_step
            torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        timed_seconds = []
        input_pipeline_seconds = 0.0
        examples_timed = 0
        for _ in range(args.steps):
            fetch_started = time.perf_counter()
            batch = next(batch_iterator)
            fetch_seconds = time.perf_counter() - fetch_started
            if world_size > 1:
                fetch_tensor = torch.tensor(fetch_seconds, dtype=torch.float64, device=device)
                dist.all_reduce(fetch_tensor, op=dist.ReduceOp.MAX)
                fetch_seconds = float(fetch_tensor.item())
            input_pipeline_seconds += fetch_seconds
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = train_one_epoch(
                model=loss_model,
                loader=[batch],
                optimizers=optimizers,
                optimizer_base_lrs=optimizer_base_lrs,
                device=device,
                carry=carry,
                optimizer_step=optimizer_step,
                total_steps=total_steps,
                warmup_steps=int(config["optimization"]["lr_warmup_steps"]),
                min_lr_ratio=float(config["optimization"]["lr_min_ratio"]),
                global_batch_size=batch_size,
                accumulation_steps=1,
                gradient_clip_norm=config["optimization"].get("gradient_clip_norm"),
                rank=rank,
                world_size=world_size,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if world_size > 1:
                elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                elapsed = float(elapsed_tensor.item())
            timed_seconds.append(elapsed + fetch_seconds)
            examples_timed += int(batch[2])
            carry, optimizer_step = result.carry, result.optimizer_step

        validation_metrics = None
        if rank == 0:
            identifiers_path = Path(args.dataset) / "identifiers.json"
            puzzles_path = Path(args.dataset) / "test_puzzles.json"
            with identifiers_path.open("r", encoding="utf-8") as stream:
                identifiers = json.load(stream)
            with puzzles_path.open("r", encoding="utf-8") as stream:
                test_puzzles = json.load(stream)
        else:
            identifiers = test_puzzles = None
        if world_size > 1:
            values = [identifiers, test_puzzles] if rank == 0 else [None, None]
            dist.broadcast_object_list(values, src=0, group=cpu_group)
            identifiers, test_puzzles = values
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

        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()
        checkpoint_seconds = 0.0
        if rank == 0:
            checkpoint_payload = {
                "model": tree_map(loss_model.state_dict(), lambda tensor: tensor.detach().cpu()),
                "optimizers": [
                    tree_map(optimizer.state_dict(), lambda tensor: tensor.detach().cpu())
                    for optimizer in optimizers
                ],
                "benchmark_only": True,
            }
            checkpoint_seconds = _atomic_torch_save(
                checkpoint_payload, output_dir / "benchmark_checkpoint.pt"
            )
        if world_size > 1:
            elapsed_tensor = torch.tensor(checkpoint_seconds, dtype=torch.float64, device=device)
            dist.broadcast(elapsed_tensor, src=0)
            checkpoint_seconds = float(elapsed_tensor.item())

        allocated_memory = int(torch.cuda.max_memory_allocated(device))
        reserved_memory = int(torch.cuda.max_memory_reserved(device))
        if world_size > 1:
            memory_tensor = torch.tensor(
                [allocated_memory, reserved_memory], dtype=torch.int64, device=device
            )
            dist.all_reduce(memory_tensor, op=dist.ReduceOp.MAX)
            allocated_memory, reserved_memory = [int(value) for value in memory_tensor.cpu().tolist()]

        if rank == 0:
            mean_step = sum(timed_seconds) / len(timed_seconds)
            examples_per_epoch_estimate = (
                _train_dataset.metadata.total_groups
                * _train_dataset.metadata.mean_puzzle_examples
            )
            estimated_updates_per_epoch = max(1, int(examples_per_epoch_estimate / batch_size))
            estimated_train_epoch_seconds = mean_step * estimated_updates_per_epoch
            validation_seconds = float(validation_metrics["validation_inference_seconds"])
            eval_interval = int(config["run"]["eval_every_epochs"])
            checkpoint_interval = int(config["run"]["checkpoint_every_epochs"])
            validation_count_100 = math.ceil(100 / eval_interval)
            validation_count_1000 = math.ceil(1000 / eval_interval)
            checkpoint_count_100 = math.ceil(100 / checkpoint_interval) + 1
            checkpoint_count_1000 = math.ceil(1000 / checkpoint_interval) + 1
            total_100 = (
                estimated_train_epoch_seconds * 100
                + validation_seconds * validation_count_100
                + checkpoint_seconds * checkpoint_count_100
            )
            total_1000 = (
                estimated_train_epoch_seconds * 1000
                + validation_seconds * validation_count_1000
                + checkpoint_seconds * checkpoint_count_1000
            )
            try:
                mamba_version = importlib.metadata.version("mamba-ssm")
            except importlib.metadata.PackageNotFoundError:
                mamba_version = "unknown"
            try:
                conv_version = importlib.metadata.version("causal-conv1d")
            except importlib.metadata.PackageNotFoundError:
                conv_version = "unknown"
            report = {
                "status": "measured",
                "environment": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                    "world_size": world_size,
                    "mamba_ssm": mamba_version,
                    "causal_conv1d": conv_version,
                },
                "configuration": {
                    "operator": config["operator"],
                    "hidden_size": config["model"]["hidden_size"],
                    "seq_len": validation_dataset.metadata.seq_len,
                    "puzzle_prefix_len": config["model"]["puzzle_emb_len"],
                    "H_cycles": config["model"]["H_cycles"],
                    "L_cycles": config["model"]["L_cycles"],
                    "layers": config["model"]["L_layers"],
                    "mamba": {
                        key: config["model"][key]
                        for key in (
                            "mamba_d_state",
                            "mamba_expand",
                            "mamba_headdim",
                            "mamba_d_conv",
                            "mamba_d_ssm",
                            "mamba_ngroups",
                            "mamba_rmsnorm",
                            "mamba_norm_before_gate",
                            "mamba_chunk_size",
                            "mamba_use_mem_eff_path",
                            "mamba_bias",
                            "mamba_conv_bias",
                        )
                    },
                    **batch_plan,
                    "warmup_steps": args.warmup,
                    "measured_steps": args.steps,
                },
                "warmup_seconds": warmup_seconds,
                "steady_state_step_seconds": mean_step,
                "steady_state_examples_per_second": examples_timed / max(1e-9, sum(timed_seconds)),
                "input_pipeline_seconds_total": input_pipeline_seconds,
                "timed_examples": examples_timed,
                "validation": validation_metrics,
                "checkpoint_write_seconds": checkpoint_seconds,
                "checkpoint_path": str(output_dir / "benchmark_checkpoint.pt"),
                "peak_gpu_memory_allocated_bytes": allocated_memory,
                "peak_gpu_memory_reserved_bytes": reserved_memory,
                "runtime_estimate_assumptions": {
                    "examples_per_epoch_from_dataset_metadata": examples_per_epoch_estimate,
                    "optimizer_updates_per_epoch_estimate": estimated_updates_per_epoch,
                    "training_seconds_per_epoch_estimate": estimated_train_epoch_seconds,
                    "validation_seconds_per_pass_measured": validation_seconds,
                    "checkpoint_seconds_per_write_measured": checkpoint_seconds,
                    "validation_passes_100_epoch_estimate": validation_count_100,
                    "validation_passes_1000_epoch_estimate": validation_count_1000,
                    "checkpoint_writes_100_epoch_estimate_including_initial": checkpoint_count_100,
                    "checkpoint_writes_1000_epoch_estimate_including_initial": checkpoint_count_1000,
                    "total_seconds_100_epoch_estimate": total_100,
                    "total_seconds_1000_epoch_estimate": total_1000,
                    "limitations": "Assumes measured steady-state step and current validation/checkpoint costs remain stable; examples-per-epoch is metadata-based while the sampler can drop partial batches and truncate a group at a batch boundary. No extrapolated estimate is trusted until this full-configuration GPU benchmark is run.",
                },
            }
            report_path = output_dir / "benchmark.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, sort_keys=True), flush=True)
        return
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
