"""Build upstream TRM arrays from the ARC Prize 2026 Kaggle file layout.

The adapter changes filenames only. It uses temporary symbolic/hard links and
delegates all parsing, augmentation, tokenization, and metadata generation to
the unmodified upstream builder.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path(
    os.environ.get(
        "TRM_UPSTREAM_ROOT", PROJECT_ROOT / "external" / "TinyRecursiveModels"
    )
).expanduser().resolve()

KAGGLE_TO_UPSTREAM = {
    "arc-agi_training_challenges.json": "arc-agi_training2_challenges.json",
    "arc-agi_training_solutions.json": "arc-agi_training2_solutions.json",
    "arc-agi_evaluation_challenges.json": "arc-agi_evaluation2_challenges.json",
    "arc-agi_evaluation_solutions.json": "arc-agi_evaluation2_solutions.json",
}


def _link_without_copy(source: Path, destination: Path) -> None:
    try:
        destination.symlink_to(source)
        return
    except OSError as symlink_error:
        try:
            os.link(source, destination)
            return
        except OSError as hardlink_error:
            raise RuntimeError(
                f"Could not link {source} into the temporary compatibility layout. "
                f"Symlink error: {symlink_error}; hard-link error: {hardlink_error}."
            ) from hardlink_error


def build_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    num_aug: int = 1000,
    seed: int = 42,
    concept_dir: Path | None = None,
) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    missing = [name for name in KAGGLE_TO_UPSTREAM if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing ARC Prize 2026 file(s) in {source_dir}: {', '.join(missing)}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (UPSTREAM_ROOT / "dataset" / "build_arc_dataset.py").is_file():
        raise FileNotFoundError(
            f"TinyRecursiveModels source not found at {UPSTREAM_ROOT}. Initialize the "
            "pinned submodule with `git submodule update --init --recursive "
            "external/TinyRecursiveModels`, or set TRM_UPSTREAM_ROOT to an attached "
            "copy of the pinned checkout."
        )
    if str(UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_ROOT))
    from dataset.build_arc_dataset import DataProcessConfig, convert_dataset

    subsets = ["training2", "evaluation2"]
    with tempfile.TemporaryDirectory(prefix="trm-arc-layout-", dir=output_dir.parent) as temp:
        compatibility_dir = Path(temp)
        for kaggle_name, upstream_name in KAGGLE_TO_UPSTREAM.items():
            _link_without_copy(source_dir / kaggle_name, compatibility_dir / upstream_name)

        if concept_dir is not None:
            concept_dir = concept_dir.resolve()
            for suffix in ("challenges", "solutions"):
                name = f"arc-agi_concept_{suffix}.json"
                source = concept_dir / name
                if not source.is_file():
                    raise FileNotFoundError(f"Missing ConceptARC file: {source}")
                _link_without_copy(source, compatibility_dir / name)
            subsets.append("concept")

        convert_dataset(
            DataProcessConfig(
                input_file_prefix=str(compatibility_dir / "arc-agi"),
                output_dir=str(output_dir),
                subsets=subsets,
                test_set_name="evaluation2",
                seed=seed,
                num_aug=num_aug,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the official ARC Prize 2026 JSON files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concept-dir",
        type=Path,
        help="Optional directory containing arc-agi_concept_{challenges,solutions}.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        args.source_dir,
        args.output_dir,
        num_aug=args.num_aug,
        seed=args.seed,
        concept_dir=args.concept_dir,
    )


if __name__ == "__main__":
    main()
