"""Project-owned per-step operators for the upstream TRM reasoning loop."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


class MambaDependencyError(RuntimeError):
    """Raised when the optional CUDA-backed Mamba package is unavailable."""


def _upstream_root() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "TinyRecursiveModels"


def _load_upstream_components():
    root = _upstream_root()
    if not (root / "models" / "layers.py").is_file():
        raise FileNotFoundError(f"TinyRecursiveModels checkout not found at {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from models.layers import Attention, SwiGLU, rms_norm

    return Attention, SwiGLU, rms_norm


def load_mamba2_class():
    """Import Mamba2 only at construction time, never during project import."""

    try:
        module = importlib.import_module("mamba_ssm.modules.mamba2")
        return module.Mamba2
    except Exception as exc:
        raise MambaDependencyError(
            "operator='mamba2_attention' requires a working mamba-ssm install "
            "and its causal-conv1d dependency. On Kaggle, attach the matching "
            "offline wheels and follow docs/KAGGLE_MAMBA_INSTALL.md. Original "
            f"import error: {type(exc).__name__}: {exc}"
        ) from exc


class Mamba2AttentionBlock(nn.Module):
    """Mamba-2, Mamba-2, bidirectional attention, SwiGLU with post-norm.

    Every sublayer has its own residual followed by the upstream RMSNorm
    function. The injected Mamba class is a CPU-test seam; production callers
    omit it and resolve the optional dependency lazily.
    """

    def __init__(
        self,
        config: Any,
        *,
        mamba_cls: Callable[..., nn.Module] | None = None,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
    ) -> None:
        super().__init__()
        Attention, SwiGLU, rms_norm = _load_upstream_components()

        self.hidden_size = int(config.hidden_size)
        self.norm_eps = float(config.rms_norm_eps)
        self.forward_dtype = getattr(torch, str(config.forward_dtype))
        if self.hidden_size % headdim != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by Mamba headdim ({headdim})"
            )
        if d_state <= 0 or headdim <= 0 or expand <= 0:
            raise ValueError("Mamba d_state, headdim, and expand must all be positive")

        if mamba_cls is None:
            mamba_cls = load_mamba2_class()
        mamba_kwargs = {
            "d_model": self.hidden_size,
            "d_state": d_state,
            "headdim": headdim,
            "expand": expand,
            "dtype": self.forward_dtype,
        }
        self.mamba1 = mamba_cls(**mamba_kwargs)
        self.mamba2 = mamba_cls(**mamba_kwargs)
        self.self_attn = Attention(
            hidden_size=self.hidden_size,
            head_dim=self.hidden_size // int(config.num_heads),
            num_heads=int(config.num_heads),
            num_key_value_heads=int(config.num_heads),
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=self.hidden_size,
            expansion=float(config.expansion),
        )
        self._rms_norm = rms_norm

    def _post_norm_residual(
        self, hidden_states: torch.Tensor, update: torch.Tensor, *, name: str
    ) -> torch.Tensor:
        if update.shape != hidden_states.shape:
            raise RuntimeError(
                f"{name} changed operator shape from {tuple(hidden_states.shape)} "
                f"to {tuple(update.shape)}"
            )
        if update.device != hidden_states.device:
            raise RuntimeError(f"{name} changed operator device")
        if update.dtype != hidden_states.dtype:
            raise RuntimeError(f"{name} changed operator dtype")
        return self._rms_norm(
            hidden_states + update, variance_epsilon=self.norm_eps
        )

    def forward(
        self, cos_sin=None, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hidden_states is None:
            raise TypeError("hidden_states is required")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Mamba2AttentionBlock expects [B, S, D] with "
                f"D={self.hidden_size}; got {tuple(hidden_states.shape)}"
            )

        hidden_states = self._post_norm_residual(
            hidden_states, self.mamba1(hidden_states), name="Mamba-2 sublayer 1"
        )
        hidden_states = self._post_norm_residual(
            hidden_states, self.mamba2(hidden_states), name="Mamba-2 sublayer 2"
        )
        hidden_states = self._post_norm_residual(
            hidden_states,
            self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            name="attention sublayer",
        )
        hidden_states = self._post_norm_residual(
            hidden_states, self.mlp(hidden_states), name="MLP sublayer"
        )
        return hidden_states
