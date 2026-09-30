"""Print a non-mutating JSON inventory for the Kaggle/runtime environment."""

from __future__ import annotations

import importlib.metadata
import json
import platform


PACKAGES = (
    "torch",
    "adam-atan2",
    "einops",
    "tqdm",
    "coolname",
    "pydantic",
    "argdantic",
    "wandb",
    "omegaconf",
    "hydra-core",
    "huggingface-hub",
    "packaging",
    "ninja",
    "wheel",
    "setuptools",
    "setuptools-scm",
    "pydantic-core",
    "numba",
    "triton",
    "numpy",
    "PyYAML",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    report: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGES},
    }

    try:
        import torch

        report["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - diagnostic script
        report["torch_runtime_error"] = repr(exc)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
