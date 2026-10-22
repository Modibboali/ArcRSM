"""Thin operator factory around the unmodified upstream TRM implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_OPERATORS = (
    "attention",
    "mamba2_attention",
    "mamba2_mlp",
    "experimental_2d_ssm",
)

_MAMBA_REQUIRED_CONFIG = (
    "mamba_d_state",
    "mamba_headdim",
    "mamba_expand",
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


def upstream_root() -> Path:
    configured_root = os.environ.get("TRM_UPSTREAM_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "external" / "TinyRecursiveModels"


def _load_model_class():
    root = upstream_root()
    if not (root / "models" / "recursive_reasoning" / "trm.py").is_file():
        raise FileNotFoundError(
            f"TinyRecursiveModels source not found at {root}. This repository keeps "
            "the pinned source under external/TinyRecursiveModels; for an attached "
            "offline copy, set TRM_UPSTREAM_ROOT to the directory containing models/."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    return TinyRecursiveReasoningModel_ACTV1


def build_trm(
    config: Mapping[str, Any],
    *,
    operator: str = "attention",
    mamba_cls=None,
):
    """Build upstream TRM and optionally replace only its per-step block class.

    ``mamba_cls`` exists for architecture tests. Production callers omit it and
    load the real Mamba-2 implementation only when constructing that operator.
    """

    if operator not in SUPPORTED_OPERATORS:
        supported = ", ".join(SUPPORTED_OPERATORS)
        raise ValueError(f"Unknown operator {operator!r}; choose one of: {supported}")
    if operator == "experimental_2d_ssm":
        raise NotImplementedError(
            "operator='experimental_2d_ssm' is reserved and has not been implemented"
        )
    if operator == "mamba2_mlp":
        missing = [key for key in _MAMBA_REQUIRED_CONFIG if key not in config]
        if missing:
            raise ValueError(
                "operator='mamba2_mlp' is missing Mamba-2 configuration value(s): "
                + ", ".join(missing)
                + ". Add the qualified mamba_* settings to the model config."
            )
    model = _load_model_class()(dict(config))
    model._arc_agi_trm_metadata = {  # type: ignore[attr-defined]
        "operator": operator,
        "model": dict(config),
    }
    if operator == "attention":
        return model

    if config.get("mlp_t", False):
        raise ValueError(
            f"operator={operator!r} requires the attention-mixer TRM setting mlp_t=False"
        )

    if operator == "mamba2_attention":
        from .operators import Mamba2AttentionBlock

        block_cls = Mamba2AttentionBlock
    else:
        from .operators import Mamba2MLPBlock

        block_cls = Mamba2MLPBlock

    hidden_config = model.config
    mamba_config = {
        "d_state": int(config.get("mamba_d_state", 128)),
        "headdim": int(config.get("mamba_headdim", 64)),
        "expand": int(config.get("mamba_expand", 2)),
        "d_conv": int(config.get("mamba_d_conv", 4)),
        "d_ssm": config.get("mamba_d_ssm", None),
        "ngroups": int(config.get("mamba_ngroups", 1)),
        "rmsnorm": bool(config.get("mamba_rmsnorm", True)),
        "norm_before_gate": bool(config.get("mamba_norm_before_gate", False)),
        "chunk_size": int(config.get("mamba_chunk_size", 256)),
        "use_mem_eff_path": bool(config.get("mamba_use_mem_eff_path", True)),
        "bias": bool(config.get("mamba_bias", False)),
        "conv_bias": bool(config.get("mamba_conv_bias", True)),
    }
    model.inner.L_level.layers = model.inner.L_level.layers.__class__(
        [
            block_cls(
                hidden_config,
                mamba_cls=mamba_cls,
                **mamba_config,
            )
            for _ in range(hidden_config.L_layers)
        ]
    )
    if len(model.inner.L_level.layers) != hidden_config.L_layers or any(
        not isinstance(layer, block_cls)
        for layer in model.inner.L_level.layers
    ):
        raise RuntimeError("Mamba-2 hybrid block installation did not complete")
    return model
