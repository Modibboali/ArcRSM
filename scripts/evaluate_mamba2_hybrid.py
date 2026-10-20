"""Evaluate a saved Mamba-2 hybrid checkpoint on the task-level validation split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arc_agi_trm.experiment import (  # noqa: E402
    batch_sizes,
    checkpoint_fingerprint,
    load_experiment_config,
    manifest_fingerprint,
    seed_everything,
)
from arc_agi_trm.runner import (  # noqa: E402
    check_dataset_config,
    create_hybrid_model,
    create_loaders,
    evaluate_validation,
    load_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="Single-GPU global batch size")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Checkpoint evaluation requires the qualified CUDA Mamba-2 environment; "
            "no CUDA device is available here."
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    config = load_experiment_config(args.config)
    dataset_dir = args.dataset.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest = load_dataset_manifest(dataset_dir)
    check_dataset_config(manifest, config, allow_small=args.allow_small_dataset)
    config["dataset_manifest_fingerprint"] = manifest_fingerprint(manifest)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_fingerprint") != checkpoint_fingerprint(config):
        raise ValueError("Checkpoint architecture/data/optimizer settings do not match this config")

    train_loader, train_dataset, validation_loader, validation_dataset = create_loaders(
        dataset_dir,
        seed=int(config["seed"]),
        global_batch_size=args.batch_size,
        rank=0,
        world_size=1,
    )
    del train_loader
    del train_dataset
    plan = batch_sizes(args.batch_size, 1, 1)
    seed_everything(int(config["seed"]))
    loss_model = create_hybrid_model(
        config,
        validation_dataset.metadata,
        per_device_batch_size=plan["per_device_batch_size"],
        device=device,
    )
    loss_model.load_state_dict(checkpoint["model"], strict=True)
    with (dataset_dir / "identifiers.json").open("r", encoding="utf-8") as stream:
        identifiers = json.load(stream)
    with (dataset_dir / "test_puzzles.json").open("r", encoding="utf-8") as stream:
        test_puzzles = json.load(stream)

    started = time.perf_counter()
    metrics = evaluate_validation(
        loss_model=loss_model,
        loader=validation_loader,
        identifiers=identifiers,
        test_puzzles=test_puzzles,
        device=device,
    )
    assert metrics is not None
    metrics.update(
        checkpoint_path=str(checkpoint_path),
        checkpoint_epoch=int(checkpoint.get("completed_epochs", -1)),
        seed=int(config["seed"]),
        operator=config["operator"],
        evaluation_wall_seconds=time.perf_counter() - started,
        per_device_batch_size=plan["per_device_batch_size"],
        effective_global_batch_size=plan["effective_global_batch_size"],
    )
    output_path = args.output.expanduser().resolve() if args.output else checkpoint_path.parent / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
