"""TRM data, model, optimizer, and ARC validation adapters for experiments."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .experiment import exact_task_metrics
from .upstream import build_trm, upstream_root


def load_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    path = Path(dataset_dir).expanduser().resolve()
    if not (path / "dataset_manifest.json").is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found in {path}. Build the training/validation "
            "arrays with scripts/build_arc_agi2_from_kaggle.py first."
        )
    with (path / "dataset_manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    required_files = [
        path / relative
        for relative in (
            "train/dataset.json",
            "train/all__inputs.npy",
            "train/all__labels.npy",
            "train/all__puzzle_identifiers.npy",
            "train/all__puzzle_indices.npy",
            "train/all__group_indices.npy",
            "test/dataset.json",
            "test/all__inputs.npy",
            "test/all__labels.npy",
            "test/all__puzzle_identifiers.npy",
            "test/all__puzzle_indices.npy",
            "test/all__group_indices.npy",
            "identifiers.json",
            "test_puzzles.json",
        )
    ]
    missing = [str(file) for file in required_files if not file.is_file()]
    if missing:
        raise FileNotFoundError(
            "Dataset build is incomplete; missing required file(s): " + ", ".join(missing)
        )
    if manifest.get("validation_protocol", {}).get("official_evaluation_or_private_test_data_used"):
        raise ValueError("Dataset manifest indicates non-training evaluation data was used")
    train_ids = set(manifest.get("training_task_ids", []))
    val_ids = set(manifest.get("validation_task_ids", []))
    original_tasks = int(manifest.get("original_public_training_tasks", 0))
    if (
        not train_ids
        or not val_ids
        or train_ids.intersection(val_ids)
        or len(train_ids) != int(manifest.get("training_original_tasks", -1))
        or len(val_ids) != int(manifest.get("validation_original_tasks", -1))
        or len(train_ids) + len(val_ids) != original_tasks
    ):
        raise ValueError("Dataset manifest has missing, empty, or overlapping task-level train/validation IDs")
    return manifest


def check_dataset_config(
    manifest: Mapping[str, Any], config: Mapping[str, Any], *, allow_small: bool = False
) -> None:
    requested_aug = int(config["data"]["num_aug"])
    requested_val = int(config["data"]["validation_tasks"])
    augmentation = manifest.get("augmentation", {})
    if int(augmentation.get("requested_unique_variants_per_training_task", -1)) != requested_aug:
        raise ValueError(
            "Dataset augmentation setting does not match the experiment configuration: "
            f"built={augmentation.get('requested_unique_variants_per_training_task')}, requested={requested_aug}"
        )
    if int(manifest["validation_original_tasks"]) != requested_val:
        raise ValueError(
            f"Dataset has {manifest['validation_original_tasks']} validation tasks; "
            f"configuration requires {requested_val}"
        )
    requested_validation_variants = int(config["data"].get("validation_variants_per_task", 1))
    validation_protocol = manifest.get("validation_protocol", {})
    if int(validation_protocol.get("validation_variants_per_task", -1)) != requested_validation_variants:
        raise ValueError("Dataset validation variant count does not match the experiment configuration")
    if int(manifest["original_public_training_tasks"]) != 1000 and not allow_small:
        raise ValueError(
            "Production runs require all 1,000 public ARC-AGI-2 training tasks; "
            "use --allow-small-dataset only for a deliberate development fixture"
        )
    actual = augmentation.get("actual_variants_per_training_task", {})
    if set(actual) != set(manifest.get("training_task_ids", [])):
        raise ValueError("Dataset augmentation manifest does not enumerate every training task")
    if any(int(count) < 1 or int(count) > requested_aug + 1 for count in actual.values()):
        raise ValueError("Dataset contains an invalid per-task augmentation count")


def create_loaders(
    dataset_dir: str | Path,
    *,
    seed: int,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    root = upstream_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    except Exception as exc:
        raise RuntimeError(
            "Could not import the upstream TRM dataset loader. Check that the pinned "
            f"source exists at {root}. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    shared = {
        "seed": seed,
        "dataset_paths": [str(Path(dataset_dir).expanduser().resolve())],
        "global_batch_size": global_batch_size,
        "epochs_per_iter": 1,
        "rank": rank,
        "num_replicas": world_size,
    }
    train_dataset = PuzzleDataset(
        PuzzleDatasetConfig(test_set_mode=False, **shared), split="train"
    )
    validation_dataset = PuzzleDataset(
        PuzzleDatasetConfig(test_set_mode=True, **shared), split="test"
    )
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=None, num_workers=0)
    if train_dataset.metadata.seq_len != 900 or train_dataset.metadata.vocab_size != 12:
        raise ValueError(
            "ARC metadata mismatch: expected seq_len=900 and vocab_size=12, got "
            f"seq_len={train_dataset.metadata.seq_len}, vocab_size={train_dataset.metadata.vocab_size}"
        )
    if train_dataset.metadata.num_puzzle_identifiers != validation_dataset.metadata.num_puzzle_identifiers:
        raise ValueError("Train and validation datasets have different puzzle identifier tables")
    return train_loader, train_dataset, validation_loader, validation_dataset


def create_experiment_model(
    config: Mapping[str, Any],
    metadata,
    *,
    per_device_batch_size: int,
    device: torch.device,
    mamba_cls=None,
):
    from models.losses import ACTLossHead

    model_config = dict(config["model"])
    model_config.update(
        batch_size=per_device_batch_size,
        seq_len=int(metadata.seq_len),
        vocab_size=int(metadata.vocab_size),
        num_puzzle_identifiers=int(metadata.num_puzzle_identifiers),
        causal=False,
    )
    with torch.device(device):
        trm = build_trm(
            model_config,
            operator=str(config["operator"]),
            mamba_cls=mamba_cls,
        )
        loss_model = ACTLossHead(trm, loss_type="stablemax_cross_entropy")

    operator = str(config["operator"])
    layers = trm.inner.L_level.layers
    if len(layers) != int(model_config["L_layers"]):
        raise RuntimeError("Model factory installed an unexpected number of shared TRM blocks")
    if operator == "mamba2_attention":
        from .operators import Mamba2AttentionBlock

        expected_cls = Mamba2AttentionBlock
    elif operator == "mamba2_mlp":
        from .operators import Mamba2MLPBlock

        expected_cls = Mamba2MLPBlock
    else:
        expected_cls = None
    if expected_cls is not None:
        if any(not isinstance(layer, expected_cls) for layer in layers):
            raise RuntimeError(f"Model factory did not install the configured {operator} blocks")
        parameter_ids = {id(parameter) for parameter in loss_model.parameters()}
        mamba_modules = [
            getattr(layer, "mamba", None)
            for layer in layers
        ] if operator == "mamba2_mlp" else [
            mamba
            for layer in layers
            for mamba in (layer.mamba1, layer.mamba2)
        ]
        expected_mamba_parameters = {
            id(parameter)
            for mamba in mamba_modules
            for parameter in mamba.parameters()
        }
        if not expected_mamba_parameters or not expected_mamba_parameters.issubset(parameter_ids):
            raise RuntimeError("One or more Mamba-2 parameters are absent from model.parameters()")
    return loss_model


# Kept for existing experiment scripts and tests.
create_hybrid_model = create_experiment_model


def create_optimizers(
    loss_model,
    config: Mapping[str, Any],
    *,
    world_size: int,
    dense_optimizer_cls=None,
):
    try:
        from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
    except Exception as exc:
        raise RuntimeError(
            "Training requires the upstream sparse SignSGD optimizer. "
            "Check the pinned TRM source. Original "
            f"import error: {type(exc).__name__}: {exc}"
        ) from exc
    if dense_optimizer_cls is None:
        try:
            from adam_atan2 import AdamATan2
        except Exception as exc:
            raise RuntimeError(
                "Training requires the upstream AdamATan2 optimizer. Install the "
                "already-qualified Kaggle training requirements. Original "
                f"import error: {type(exc).__name__}: {exc}"
            ) from exc
        dense_optimizer_cls = AdamATan2
    settings = config["optimization"]
    sparse_embedding = loss_model.model.puzzle_emb
    sparse_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
        sparse_embedding.buffers(),
        lr=0,
        weight_decay=float(settings["puzzle_emb_weight_decay"]),
        world_size=world_size,
    )
    dense_optimizer = dense_optimizer_cls(
        loss_model.parameters(),
        lr=0,
        weight_decay=float(settings["weight_decay"]),
        betas=(float(settings["beta1"]), float(settings["beta2"])),
    )
    optimized_parameter_ids = {
        id(parameter)
        for group in dense_optimizer.param_groups
        for parameter in group["params"]
    }
    missing = [name for name, parameter in loss_model.named_parameters() if id(parameter) not in optimized_parameter_ids]
    if missing:
        raise RuntimeError(f"Optimizer is missing trainable model parameters: {missing[:8]}")
    return [sparse_optimizer, dense_optimizer], [
        float(settings["puzzle_emb_lr"]),
        float(settings["lr"]),
    ]


def _crop_tokens(tokens: np.ndarray) -> np.ndarray:
    grid = np.asarray(tokens).reshape(30, 30)
    max_area = 0
    max_height = 0
    max_width = 0
    width = 30
    for row in range(30):
        for column in range(width):
            value = int(grid[row, column])
            if value < 2 or value > 11:
                width = column
                break
        area = (row + 1) * width
        if area > max_area:
            max_area = area
            max_height = row + 1
            max_width = width
    return (grid[:max_height, :max_width] - 2).astype(np.uint8)


def _grid_hash(grid: np.ndarray) -> str:
    import hashlib

    grid = np.asarray(grid, dtype=np.uint8)
    return hashlib.sha256(
        bytes((grid.shape[0], grid.shape[1])) + grid.tobytes(order="C")
    ).hexdigest()


def _validation_expected(test_puzzles: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
    expected = {}
    for task_id, task in test_puzzles.items():
        expected[task_id] = [
            (
                _grid_hash(np.asarray(pair["input"], dtype=np.uint8)),
                _grid_hash(np.asarray(pair["output"], dtype=np.uint8)),
            )
            for pair in task["test"]
        ]
    return expected


def evaluate_validation(
    *,
    loss_model,
    loader,
    identifiers: list[str],
    test_puzzles: Mapping[str, Any],
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    cpu_group=None,
) -> dict[str, Any] | None:
    model_was_training = loss_model.training
    loss_model.eval()
    local = {"loss_sum": 0.0, "examples": 0, "predictions": [], "seconds": 0.0}
    start = time.perf_counter()
    with torch.inference_mode():
        for _set_name, source_batch, _global_effective in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in source_batch.items()}
            with torch.device(device):
                carry = loss_model.initial_carry(batch)
            while True:
                carry, _raw_loss, _metrics, outputs, all_finished = loss_model(
                    carry=carry,
                    batch=batch,
                    return_keys=("logits", "preds", "q_halt_logits"),
                )
                if bool(all_finished):
                    break

            labels = carry.current_data["labels"]
            mask = labels != -100
            valid = mask.any(dim=-1)
            counts = mask.sum(dim=-1).clamp_min(1)
            token_losses = loss_model.loss_fn(
                outputs["logits"], labels, ignore_index=-100, valid_mask=mask
            )
            lm_per_example = (token_losses / counts.unsqueeze(-1)).sum(dim=-1)
            predictions = outputs["preds"]
            correct_tokens = mask & (predictions == labels)
            sequence_correct = correct_tokens.sum(dim=-1) == mask.sum(dim=-1)
            q_losses = F.binary_cross_entropy_with_logits(
                outputs["q_halt_logits"],
                sequence_correct.to(outputs["q_halt_logits"].dtype),
                reduction="none",
            )
            valid_float = valid.to(lm_per_example.dtype)
            local["loss_sum"] += float(
                (lm_per_example * valid_float + 0.5 * q_losses * valid_float).sum().item()
            )
            local["examples"] += int(valid.sum().item())

            inputs_cpu = carry.current_data["inputs"].detach().cpu().numpy()
            predictions_cpu = predictions.detach().cpu().numpy()
            ids_cpu = carry.current_data["puzzle_identifiers"].detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()
            for identifier, input_tokens, output_tokens, is_valid in zip(
                ids_cpu, inputs_cpu, predictions_cpu, valid_cpu
            ):
                if not is_valid or int(identifier) == 0:
                    continue
                task_id = identifiers[int(identifier)]
                input_hash = _grid_hash(_crop_tokens(input_tokens))
                output_hash = _grid_hash(_crop_tokens(output_tokens))
                local["predictions"].append((task_id, input_hash, output_hash))
            del carry, batch, outputs

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    local["seconds"] = time.perf_counter() - start
    if model_was_training:
        loss_model.train()

    if world_size > 1:
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local, gathered, dst=0, group=cpu_group)
        if rank != 0:
            return None
        parts = gathered
    else:
        parts = [local]

    merged_predictions = [item for part in parts for item in part["predictions"]]
    metrics = exact_task_metrics(_validation_expected(test_puzzles), merged_predictions)
    loss_sum = sum(float(part["loss_sum"]) for part in parts)
    examples = sum(int(part["examples"]) for part in parts)
    elapsed = max(float(part["seconds"]) for part in parts)
    return {
        "validation_loss": loss_sum / max(1, examples),
        "validation_tasks_evaluated": metrics["tasks_evaluated"],
        "validation_tasks_solved": metrics["tasks_solved"],
        "validation_task_exact_match": metrics["task_exact_match"],
        "validation_grid_exact_accuracy": metrics["grid_exact_accuracy"],
        "validation_grid_examples": metrics["grid_examples"],
        "validation_grid_examples_solved": metrics["grid_examples_solved"],
        "validation_examples_in_loss": examples,
        "validation_attempts": 1,
        "validation_protocol": "one greedy top-1 output per public training-task test input; a task is solved only when every test output is exact",
        "validation_inference_seconds": elapsed,
    }

