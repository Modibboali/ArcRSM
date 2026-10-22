"""Evaluate a model-only artifact or trusted training checkpoint on ARC validation."""

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
from arc_agi_trm.checkpoints import (  # noqa: E402
    MODEL_CHECKPOINT_KIND,
    load_checkpoint,
    load_pretrained_model,
)
from arc_agi_trm.runner import (  # noqa: E402
    check_dataset_config,
    create_experiment_model,
    create_loaders,
    evaluate_validation,
    load_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Experiment config; defaults to the artifact snapshot")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="Single-GPU global batch size")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--trusted-full-checkpoint",
        action="store_true",
        help="Allow pickle loading of a full checkpoint created by this project's trainer",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args()

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset_dir = args.dataset.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    try:
        safe_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        safe_payload = None
    is_pretrained = isinstance(safe_payload, dict) and safe_payload.get("kind") == MODEL_CHECKPOINT_KIND
    if is_pretrained:
        artifact_config = safe_payload.get("config")
        if not isinstance(artifact_config, dict) or "model" not in artifact_config:
            raise ValueError("Model-only artifact is missing its configuration snapshot")
        config = load_experiment_config(args.config) if args.config else artifact_config
        if args.config and str(config.get("operator")) != str(safe_payload["architecture"]):
            raise ValueError(
                f"Evaluation config architecture {config.get('operator')!r} does not match "
                f"saved architecture {safe_payload['architecture']!r}"
            )
    else:
        if not args.trusted_full_checkpoint:
            raise ValueError(
                "Checkpoint is not a tensor-only model artifact. For a full checkpoint "
                "created by this project's trainer, pass --trusted-full-checkpoint."
            )
        if args.config is None:
            raise ValueError("--config is required when evaluating a full training checkpoint")
        config = load_experiment_config(args.config)
    if str(config.get("operator", "")).startswith("mamba2_") and device.type == "cpu":
        raise RuntimeError(
            "The qualified mamba-ssm Mamba-2 path uses CUDA kernels and cannot run this "
            "inference on CPU. Use --device cuda with the matching Mamba-2 and "
            "causal-conv1d wheels; no attention fallback is performed."
        )
    manifest = load_dataset_manifest(dataset_dir)
    check_dataset_config(manifest, config, allow_small=args.allow_small_dataset)
    config["dataset_manifest_fingerprint"] = manifest_fingerprint(manifest)

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
    config["model"].update(
        batch_size=plan["per_device_batch_size"],
        seq_len=int(validation_dataset.metadata.seq_len),
        vocab_size=int(validation_dataset.metadata.vocab_size),
        num_puzzle_identifiers=int(validation_dataset.metadata.num_puzzle_identifiers),
    )
    if not is_pretrained:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", trusted=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("schema_version", 1) not in {1, 2}:
            raise ValueError("Unsupported full training checkpoint schema")
        if checkpoint.get("config_fingerprint") != checkpoint_fingerprint(config):
            raise ValueError("Checkpoint architecture/data/optimizer settings do not match this config")
        if checkpoint.get("architecture") != config["operator"]:
            raise ValueError("Full checkpoint architecture does not match the evaluation config")
    seed_everything(int(config["seed"]))
    if is_pretrained:
        for key, value in config["model"].items():
            if key == "batch_size":
                continue
            if key in artifact_config["model"] and artifact_config["model"][key] != value:
                raise ValueError(
                    f"Evaluation model setting {key}={value!r} does not match the "
                    f"pretrained artifact value {artifact_config['model'][key]!r}"
                )
        loss_model = load_pretrained_model(
            checkpoint_path,
            device=device,
            batch_size=plan["per_device_batch_size"],
        )
        artifact_metadata = safe_payload.get("metadata", {})
        checkpoint_epoch = int(
            artifact_metadata.get("epoch", artifact_metadata.get("completed_epochs", -1))
        )
    else:
        loss_model = create_experiment_model(
            config,
            validation_dataset.metadata,
            per_device_batch_size=plan["per_device_batch_size"],
            device=device,
        )
        loss_model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_epoch = int(checkpoint.get("completed_epochs", -1))
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
        checkpoint_epoch=checkpoint_epoch,
        seed=int(config["seed"]),
        operator=safe_payload["architecture"] if is_pretrained else config["operator"],
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
