"""Thin operator factory around the unmodified upstream TRM implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_OPERATORS = ("attention", "mamba2_attention", "experimental_2d_ssm")


def upstream_root() -> Path:
    configured_root = os.environ.get("TRM_UPSTREAM_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "external" / "TinyRecursiveModels"


def _load_model_class():
    root = upstream_root()
    if not (root / "models" / "recursive_reasoning" / "trm.py").is_file():
        raise FileNotFoundError(
            f"TinyRecursiveModels source not found at {root}. Initialize the pinned "
            "submodule with `git submodule update --init --recursive "
            "external/TinyRecursiveModels`, or set TRM_UPSTREAM_ROOT to an attached "
            "copy of the pinned checkout (required when Kaggle internet is disabled)."
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
    model = _load_model_class()(dict(config))
    if operator == "attention":
        return model

    if config.get("mlp_t", False):
        raise ValueError(
            f"operator={operator!r} requires the attention-mixer TRM setting mlp_t=False"
        )

    from .operators import Mamba2AttentionBlock

    hidden_config = model.config
    mamba_config = {
        "d_state": int(config.get("mamba_d_state", 128)),
        "headdim": int(config.get("mamba_headdim", 64)),
        "expand": int(config.get("mamba_expand", 2)),
    }
    model.inner.L_level.layers = model.inner.L_level.layers.__class__(
        [
            Mamba2AttentionBlock(
                hidden_config,
                mamba_cls=mamba_cls,
                **mamba_config,
            )
            for _ in range(hidden_config.L_layers)
        ]
    )
    return model
