"""Export the model weights from a trusted full TRM training checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arc_agi_trm import build_trm, save_pretrained  # noqa: E402
from arc_agi_trm.checkpoints import load_checkpoint  # noqa: E402
from arc_agi_trm.upstream import SUPPORTED_OPERATORS  # noqa: E402
from arc_agi_trm.upstream import _MAMBA_REQUIRED_CONFIG  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trusted-full-checkpoint",
        action="store_true",
        required=True,
        help="Confirm this pickle-based checkpoint was created by this project's trainer",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu", trusted=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("schema_version", 1) not in {1, 2}:
        raise ValueError("Unsupported full training checkpoint schema")
    config = checkpoint.get("config")
    operator = checkpoint.get("architecture")
    if not isinstance(config, dict) or operator not in SUPPORTED_OPERATORS:
        raise ValueError("Full checkpoint does not contain a recognized model config and architecture")
    if config.get("operator") != operator:
        raise ValueError("Full checkpoint architecture and config operator disagree")
    if operator == "mamba2_mlp":
        missing = [key for key in _MAMBA_REQUIRED_CONFIG if key not in config.get("model", {})]
        if missing:
            raise ValueError("Checkpoint is missing required Mamba-2 model settings: " + ", ".join(missing))

    device = torch.device(args.device)
    with torch.device(device):
        trm = build_trm(config["model"], operator=operator)
        from models.losses import ACTLossHead

        model = ACTLossHead(trm, loss_type="stablemax_cross_entropy")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Full checkpoint has no model state")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    save_pretrained(
        model,
        args.output,
        config,
        metadata={
            "completed_epochs": int(checkpoint.get("completed_epochs", -1)),
            "global_step": int(checkpoint.get("global_step", checkpoint.get("optimizer_step", -1))),
            "best_validation": checkpoint.get("best_validation", {}),
            "exported_from": str(args.checkpoint.expanduser().resolve()),
        },
    )
    print(f"Saved model-only pretrained artifact: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
