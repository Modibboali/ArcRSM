"""Versioned model-only artifacts for TRM inference and weight initialization."""

from __future__ import annotations

import json
import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch

MODEL_CHECKPOINT_SCHEMA = 1
MODEL_CHECKPOINT_KIND = "arc_agi_trm_pretrained"


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    trusted: bool = False,
) -> dict[str, Any]:
    """Load a full training checkpoint containing Python/NumPy state.

    Full resume files use pickle for RNG tuples and TRM carry dataclasses.
    Set ``trusted=True`` only for files created by this project's trainer.
    Model-only artifacts should use :func:`load_pretrained` instead.
    """

    if not trusted:
        raise ValueError(
            "Full training checkpoints contain pickle state; pass trusted=True only "
            "for a checkpoint produced by the ArcRSM training script."
        )
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Training checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"Could not read trusted training checkpoint {source}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("checkpoint_kind") not in {
        None,
        "arc_agi_trm_training",
    }:
        raise ValueError(f"{source} is not an ArcRSM full training checkpoint")
    return payload


def _plain_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, default=str))


def source_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _unwrap_trm(model):
    candidate = model
    while not hasattr(candidate, "inner") and hasattr(candidate, "model"):
        candidate = candidate.model
    return candidate


def _model_metadata(model) -> dict[str, Any] | None:
    metadata = getattr(_unwrap_trm(model), "_arc_agi_trm_metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else None


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if "operator" not in config or "model" not in config:
        raise ValueError("Pretrained configuration must contain operator and model entries")
    operator = str(config["operator"])
    model_config = dict(config["model"])
    model_config.pop("batch_size", None)
    model_config.pop("causal", None)
    if operator == "mamba2_mlp":
        required = (
            "mamba_d_state", "mamba_headdim", "mamba_expand", "mamba_d_conv",
            "mamba_d_ssm", "mamba_ngroups", "mamba_rmsnorm",
            "mamba_norm_before_gate", "mamba_chunk_size",
            "mamba_use_mem_eff_path", "mamba_bias", "mamba_conv_bias",
        )
        missing = [key for key in required if key not in model_config]
        if missing:
            raise ValueError(
                "mamba2_mlp checkpoint config is missing required Mamba-2 setting(s): "
                + ", ".join(missing)
            )
    return {"operator": operator, "model": _plain_json(model_config)}


def _architecture_hyperparameters(config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = dict(config["model"])
    return _plain_json(
        {
            key: value
            for key, value in model_config.items()
            if key.startswith("mamba_")
            or key
            in {
                "hidden_size",
                "expansion",
                "num_heads",
                "seq_len",
                "vocab_size",
                "puzzle_emb_ndim",
                "puzzle_emb_len",
                "num_puzzle_identifiers",
                "H_cycles",
                "L_cycles",
                "H_layers",
                "L_layers",
                "pos_encodings",
                "forward_dtype",
            }
        }
    )


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def save_pretrained(
    model,
    path: str | Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save tensor-only model weights and sufficient architecture metadata."""

    destination = Path(path).expanduser()
    normalized = _normalized_config(config)
    actual = _model_metadata(model)
    if actual is not None and actual["operator"] != normalized["operator"]:
        raise ValueError(
            f"Configured architecture {normalized['operator']!r} does not match the "
            f"model architecture {actual['operator']!r}"
        )
    state = {
        key: value.detach().to(device="cpu", copy=True)
        for key, value in model.state_dict().items()
    }
    payload = {
        "kind": MODEL_CHECKPOINT_KIND,
        "schema_version": MODEL_CHECKPOINT_SCHEMA,
        "architecture": normalized["operator"],
        "architecture_hyperparameters": _architecture_hyperparameters(config),
        "model_config": normalized["model"],
        "config": _plain_json(config),
        "model_kind": "act_loss_head" if hasattr(model, "loss_fn") else "trm",
        "vocabulary": {
            "size": normalized["model"].get("vocab_size"),
            "output_size": normalized["model"].get("vocab_size"),
        },
        "metadata": _plain_json(dict(metadata or {})),
        "source_revision": source_revision(),
        "saved_unix_time": time.time(),
        "state_dict": state,
    }
    _atomic_torch_save(payload, destination)
    return destination


def _read_pretrained(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Pretrained model artifact does not exist: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Could not safely read pretrained model artifact {source}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("kind") != MODEL_CHECKPOINT_KIND:
        raise ValueError(f"{source} is not an ArcRSM model-only pretrained artifact")
    if payload.get("schema_version") != MODEL_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Unsupported pretrained checkpoint schema {payload.get('schema_version')!r} "
            f"in {source}; supported schema is {MODEL_CHECKPOINT_SCHEMA}"
        )
    required = (
        "architecture",
        "architecture_hyperparameters",
        "vocabulary",
        "model_config",
        "config",
        "state_dict",
        "model_kind",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Pretrained artifact {source} is missing required field(s): {missing}")
    if not isinstance(payload["state_dict"], dict):
        raise ValueError(f"Pretrained artifact {source} has an invalid state_dict")
    normalized = _normalized_config(payload["config"])
    if normalized["operator"] != payload["architecture"]:
        raise ValueError("Pretrained artifact architecture and saved config disagree")
    if normalized["model"] != payload["model_config"]:
        raise ValueError("Pretrained artifact model_config does not match its config snapshot")
    if _architecture_hyperparameters(payload["config"]) != payload["architecture_hyperparameters"]:
        raise ValueError("Pretrained artifact hyperparameters do not match its config snapshot")
    expected_vocab = {
        "size": normalized["model"].get("vocab_size"),
        "output_size": normalized["model"].get("vocab_size"),
    }
    if payload["vocabulary"] != expected_vocab:
        raise ValueError("Pretrained artifact vocabulary metadata does not match its model config")
    return payload


def load_pretrained(
    model,
    path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
    *,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a model-only artifact into a model; return all non-strict skips."""

    payload = _read_pretrained(path)
    actual_metadata = _model_metadata(model)
    if actual_metadata is not None and actual_metadata.get("operator") != payload["architecture"]:
        raise ValueError(
            f"Checkpoint architecture {payload['architecture']!r} is incompatible with "
            f"the constructed model architecture {actual_metadata.get('operator')!r}"
        )
    target = expected_config or _model_metadata(model)
    if target is None:
        raise ValueError(
            "Cannot validate the target model configuration. Build it with build_trm "
            "or pass expected_config explicitly."
        )
    expected = _normalized_config(target)
    if actual_metadata is not None:
        actual = _normalized_config(actual_metadata)
        if actual != expected:
            raise ValueError(
                "expected_config does not describe the constructed model: "
                + json.dumps(
                    {"constructed": actual, "expected": expected},
                    sort_keys=True,
                    default=str,
                )
            )
    saved = {"operator": payload["architecture"], "model": payload["model_config"]}
    if expected["operator"] != saved["operator"]:
        raise ValueError(
            f"Checkpoint architecture {saved['operator']!r} is incompatible with "
            f"target architecture {expected['operator']!r}"
        )
    config_differences = {
        key: {"checkpoint": saved["model"].get(key), "target": expected["model"].get(key)}
        for key in sorted(set(saved["model"]) | set(expected["model"]))
        if saved["model"].get(key) != expected["model"].get(key)
    }
    current_state = model.state_dict()
    saved_state = payload["state_dict"]
    missing = sorted(set(current_state) - set(saved_state))
    unexpected = sorted(set(saved_state) - set(current_state))
    mismatched_shapes = {
        key: {"checkpoint": tuple(saved_state[key].shape), "target": tuple(current_state[key].shape)}
        for key in sorted(set(current_state) & set(saved_state))
        if not isinstance(saved_state[key], torch.Tensor)
        or tuple(saved_state[key].shape) != tuple(current_state[key].shape)
    }
    report = {
        "architecture": saved["operator"],
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_shapes": mismatched_shapes,
        "configuration_differences": config_differences,
        "strict": bool(strict),
    }
    if strict and (missing or unexpected or mismatched_shapes or config_differences):
        raise ValueError(
            "Pretrained checkpoint is incompatible with the target model: "
            + json.dumps(report, sort_keys=True, default=str)
        )
    if missing or unexpected or mismatched_shapes or config_differences:
        warnings.warn(
            "Non-strict pretrained load skipped or differed on: "
            + json.dumps(report, sort_keys=True, default=str),
            RuntimeWarning,
            stacklevel=2,
        )
    usable_state = {
        key: value
        for key, value in saved_state.items()
        if key in current_state
        and isinstance(value, torch.Tensor)
        and tuple(value.shape) == tuple(current_state[key].shape)
    }
    model.load_state_dict(usable_state, strict=False)
    model.to(torch.device(device))
    report["metadata"] = payload.get("metadata", {})
    report["source_revision"] = payload.get("source_revision")
    return report


def load_pretrained_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    batch_size: int | None = None,
    mamba_cls=None,
):
    """Reconstruct the saved TRM architecture, load it safely, and enter eval mode."""

    payload = _read_pretrained(path)
    config = payload["config"]
    if payload["architecture"] not in {"attention", "mamba2_attention", "mamba2_mlp"}:
        raise ValueError(f"Unsupported saved architecture {payload['architecture']!r}")
    target_device = torch.device(device)
    if (
        payload["architecture"].startswith("mamba2_")
        and target_device.type == "cpu"
        and mamba_cls is None
    ):
        raise RuntimeError(
            "The qualified mamba-ssm Mamba-2 implementation requires CUDA for inference. "
            "Use a CUDA device or inject a CPU-compatible Mamba implementation explicitly; "
            "the architecture will not fall back to attention."
        )
    model_config = dict(config["model"])
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model_config["batch_size"] = int(batch_size)
    elif "batch_size" not in model_config:
        raise ValueError("Saved model config has no batch_size; pass batch_size explicitly")

    from .upstream import build_trm

    trm = build_trm(
        model_config,
        operator=str(payload["architecture"]),
        mamba_cls=mamba_cls,
    )
    if payload["model_kind"] == "act_loss_head":
        from models.losses import ACTLossHead

        model = ACTLossHead(trm, loss_type="stablemax_cross_entropy")
    elif payload["model_kind"] == "trm":
        model = trm
    else:
        raise ValueError(f"Unsupported model wrapper {payload['model_kind']!r}")
    # Runtime batch size is intentionally omitted from compatibility checks.
    load_pretrained(model, path, target_device, strict=True, expected_config=config)
    model.eval()
    return model
