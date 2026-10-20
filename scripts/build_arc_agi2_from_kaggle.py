"""Build a streamed ARC-AGI-2 train/validation dataset from Kaggle files.

Only the 1,000 public training tasks are read. A deterministic task-level
holdout is written to ``test/`` for model selection; no evaluation or private
test files are used. Training task groups retain the upstream TRM sampler's
original-plus-unique-augmentations representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path(
    os.environ.get(
        "TRM_UPSTREAM_ROOT", PROJECT_ROOT / "external" / "TinyRecursiveModels"
    )
).expanduser().resolve()

TRAINING_CHALLENGES = "arc-agi_training_challenges.json"
TRAINING_SOLUTIONS = "arc-agi_training_solutions.json"
_NPY_DATA_OFFSET = 4096
_NPY_HEADER_SIZE = _NPY_DATA_OFFSET - 10
_ARC_SEQUENCE_LENGTH = 30 * 30


class _NpyRowWriter:
    """Write a two-dimensional uint8 NPY array without holding it in RAM."""

    def __init__(self, path: Path, width: int = _ARC_SEQUENCE_LENGTH) -> None:
        self.path = path
        self.width = width
        self.rows = 0
        self._file = path.open("wb+")
        self._write_header(0)
        self._file.seek(_NPY_DATA_OFFSET)

    def append(self, rows: np.ndarray) -> None:
        rows = np.ascontiguousarray(rows, dtype=np.uint8)
        if rows.ndim != 2 or rows.shape[1] != self.width:
            raise ValueError(f"Expected rows shaped [N, {self.width}], got {rows.shape}")
        self._file.write(rows.tobytes(order="C"))
        self.rows += rows.shape[0]

    def _write_header(self, row_count: int) -> None:
        header_text = (
            "{'descr': '|u1', 'fortran_order': False, "
            f"'shape': ({row_count}, {self.width}), }}"
        )
        header = header_text.encode("latin1")
        if len(header) + 1 > _NPY_HEADER_SIZE:
            raise ValueError("Reserved NPY header is too small")
        header += b" " * (_NPY_HEADER_SIZE - len(header) - 1) + b"\n"
        self._file.seek(0)
        self._file.write(b"\x93NUMPY\x01\x00")
        self._file.write(struct.pack("<H", len(header)))
        self._file.write(header)

    def close(self) -> None:
        if self._file.closed:
            return
        self._write_header(self.rows)
        self._file.truncate(_NPY_DATA_OFFSET + self.rows * self.width)
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()

    def __enter__(self) -> "_NpyRowWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_upstream_builder():
    if not (UPSTREAM_ROOT / "dataset" / "build_arc_dataset.py").is_file():
        raise FileNotFoundError(
            f"TinyRecursiveModels source not found at {UPSTREAM_ROOT}. Initialize the "
            "pinned checkout or set TRM_UPSTREAM_ROOT to an attached copy."
        )
    root_text = str(UPSTREAM_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from dataset.build_arc_dataset import (
        convert_single_arc_puzzle,
        np_grid_to_seq_translational_augment,
    )
    from dataset.common import PuzzleDatasetMetadata

    return (
        convert_single_arc_puzzle,
        np_grid_to_seq_translational_augment,
        PuzzleDatasetMetadata,
    )


def _split_ids(task_ids: list[str], validation_tasks: int, seed: int) -> tuple[list[str], list[str]]:
    if not 0 < validation_tasks < len(task_ids):
        raise ValueError(
            f"validation_tasks must be between 1 and {len(task_ids) - 1}; got {validation_tasks}"
        )
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(task_ids), dtype=object)
    validation = set(rng.permutation(shuffled)[:validation_tasks].tolist())
    train_ids = [task_id for task_id in sorted(task_ids) if task_id not in validation]
    validation_ids = [task_id for task_id in sorted(task_ids) if task_id in validation]
    return train_ids, validation_ids


def _attach_solutions(challenges: dict[str, Any], solutions: dict[str, Any]) -> None:
    if set(challenges) != set(solutions):
        missing = sorted(set(challenges) - set(solutions))
        extra = sorted(set(solutions) - set(challenges))
        raise ValueError(
            f"Challenge/solution task IDs differ (missing solutions={missing[:3]}, "
            f"extra solutions={extra[:3]})"
        )
    for task_id, task in challenges.items():
        outputs = solutions[task_id]
        if len(outputs) != len(task["test"]):
            raise ValueError(
                f"Task {task_id} has {len(task['test'])} test inputs but "
                f"{len(outputs)} public solutions"
            )
        for example, output in zip(task["test"], outputs):
            example["output"] = output


def _write_task_group(
    *,
    group: list[Any],
    writer: _NpyRowWriter,
    labels_writer: _NpyRowWriter,
    puzzle_identifiers: list[int],
    puzzle_indices: list[int],
    identifier_names: list[str],
    group_indices: list[int],
    next_identifier: int,
    allow_translation: bool,
    np_grid_to_seq_translational_augment,
) -> tuple[int, int]:
    pair_count = 0
    for puzzle in group:
        if not puzzle.examples:
            raise ValueError(f"Task variant {puzzle.id!r} has no input/output pairs")
        untranslated_example = np.random.randint(0, len(puzzle.examples))
        encoded_inputs = []
        encoded_labels = []
        for example_index, (input_grid, output_grid) in enumerate(puzzle.examples):
            encoded_input, encoded_label = np_grid_to_seq_translational_augment(
                input_grid,
                output_grid,
                do_translation=allow_translation and example_index != untranslated_example,
            )
            encoded_inputs.append(encoded_input)
            encoded_labels.append(encoded_label)
        writer.append(np.stack(encoded_inputs))
        labels_writer.append(np.stack(encoded_labels))
        pair_count += len(encoded_inputs)
        puzzle_indices.append(writer.rows)
        puzzle_identifiers.append(next_identifier)
        identifier_names.append(puzzle.id)
        next_identifier += 1
    group_indices.append(len(puzzle_identifiers))
    return next_identifier, pair_count


def build_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    num_aug: int = 1000,
    seed: int = 42,
    validation_tasks: int = 100,
    expected_task_count: int | None = 1000,
) -> dict[str, Any]:
    """Build an ARC-AGI-2 dataset while keeping array memory usage bounded.

    ``num_aug`` is the maximum number of unique augmented variants in addition
    to each original training task. The upstream builder may produce fewer for
    highly symmetric tasks. Validation tasks are unaugmented and remain wholly
    outside the training sampler.
    """

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if num_aug < 0:
        raise ValueError("num_aug must be zero or greater")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    challenge_path = source_dir / TRAINING_CHALLENGES
    solutions_path = source_dir / TRAINING_SOLUTIONS
    for path in (challenge_path, solutions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required public ARC-AGI-2 training file is missing: {path}")

    with challenge_path.open("r", encoding="utf-8") as stream:
        challenges = json.load(stream)
    with solutions_path.open("r", encoding="utf-8") as stream:
        solutions = json.load(stream)
    if expected_task_count is not None and len(challenges) != expected_task_count:
        raise ValueError(
            f"Expected {expected_task_count} public training tasks, found {len(challenges)}. "
            "Pass expected_task_count=None only for a deliberately smaller fixture."
        )
    _attach_solutions(challenges, solutions)
    train_ids, validation_ids = _split_ids(list(challenges), validation_tasks, seed)

    (
        convert_single_arc_puzzle,
        np_grid_to_seq_translational_augment,
        PuzzleDatasetMetadata,
    ) = _load_upstream_builder()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="arc2-build-", dir=output_dir.parent))
    try:
        (temporary / "train").mkdir()
        (temporary / "test").mkdir()
        np.random.seed(seed)
        puzzle_names = ["<blank>"]
        train_puzzle_identifiers: list[int] = []
        train_puzzle_indices = [0]
        train_group_indices = [0]
        test_puzzle_identifiers: list[int] = []
        test_puzzle_indices = [0]
        test_group_indices = [0]
        train_task_variant_counts: dict[str, int] = {}
        train_grid_pairs_per_task: dict[str, int] = {}
        train_pair_total = 0
        validation_pair_total = 0
        training_original_pairs = 0
        validation_task_puzzles: dict[str, Any] = {}
        identifier = 1

        train_dir = temporary / "train"
        with _NpyRowWriter(train_dir / "all__inputs.npy") as train_inputs, _NpyRowWriter(
            train_dir / "all__labels.npy"
        ) as train_labels:
            for task_id in train_ids:
                task = challenges[task_id]
                paired_examples = task["train"] + task["test"]
                training_original_pairs += len(paired_examples)
                group_results: dict[str, Any] = {}
                convert_single_arc_puzzle(
                    group_results,
                    task_id,
                    {"train": paired_examples},
                    num_aug,
                    {"train": ("train", "all")},
                )
                variants = group_results["train"]["all"][0]
                train_task_variant_counts[task_id] = len(variants)
                train_grid_pairs_per_task[task_id] = len(paired_examples)
                identifier, pairs_written = _write_task_group(
                    group=variants,
                    writer=train_inputs,
                    labels_writer=train_labels,
                    puzzle_identifiers=train_puzzle_identifiers,
                    puzzle_indices=train_puzzle_indices,
                    identifier_names=puzzle_names,
                    group_indices=train_group_indices,
                    next_identifier=identifier,
                    allow_translation=True,
                    np_grid_to_seq_translational_augment=np_grid_to_seq_translational_augment,
                )
                train_pair_total += pairs_written

        test_dir = temporary / "test"
        with _NpyRowWriter(test_dir / "all__inputs.npy") as test_inputs, _NpyRowWriter(
            test_dir / "all__labels.npy"
        ) as test_labels:
            for task_id in validation_ids:
                task = challenges[task_id]
                validation_task_puzzles[task_id] = task
                group_results = {}
                # Validation is evaluated on each original public test input once.
                convert_single_arc_puzzle(
                    group_results,
                    task_id,
                    {"test": task["test"]},
                    0,
                    {"test": ("test", "all")},
                )
                variants = group_results["test"]["all"][0]
                identifier, pairs_written = _write_task_group(
                    group=variants,
                    writer=test_inputs,
                    labels_writer=test_labels,
                    puzzle_identifiers=test_puzzle_identifiers,
                    puzzle_indices=test_puzzle_indices,
                    identifier_names=puzzle_names,
                    group_indices=test_group_indices,
                    next_identifier=identifier,
                    allow_translation=False,
                    np_grid_to_seq_translational_augment=np_grid_to_seq_translational_augment,
                )
                validation_pair_total += pairs_written

        num_identifiers = identifier
        train_metadata = PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            vocab_size=12,
            seq_len=_ARC_SEQUENCE_LENGTH,
            num_puzzle_identifiers=num_identifiers,
            total_groups=len(train_ids),
            mean_puzzle_examples=train_pair_total / len(train_puzzle_identifiers),
            total_puzzles=len(train_puzzle_identifiers),
            sets=["all"],
        )
        test_metadata = PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            vocab_size=12,
            seq_len=_ARC_SEQUENCE_LENGTH,
            num_puzzle_identifiers=num_identifiers,
            total_groups=len(validation_ids),
            mean_puzzle_examples=validation_pair_total / len(test_puzzle_identifiers),
            total_puzzles=len(test_puzzle_identifiers),
            sets=["all"],
        )
        for split_dir, metadata in ((train_dir, train_metadata), (test_dir, test_metadata)):
            np.save(split_dir / "all__puzzle_identifiers.npy", np.asarray(
                train_puzzle_identifiers if split_dir == train_dir else test_puzzle_identifiers,
                dtype=np.int32,
            ))
            np.save(split_dir / "all__puzzle_indices.npy", np.asarray(
                train_puzzle_indices if split_dir == train_dir else test_puzzle_indices,
                dtype=np.int32,
            ))
            np.save(split_dir / "all__group_indices.npy", np.asarray(
                train_group_indices if split_dir == train_dir else test_group_indices,
                dtype=np.int32,
            ))
            with (split_dir / "dataset.json").open("w", encoding="utf-8") as stream:
                json.dump(metadata.model_dump(), stream)

        with (temporary / "identifiers.json").open("w", encoding="utf-8") as stream:
            json.dump(puzzle_names, stream)
        with (temporary / "test_puzzles.json").open("w", encoding="utf-8") as stream:
            json.dump(validation_task_puzzles, stream)

        manifest = {
            "source": "ARC Prize 2026 ARC-AGI-2 public training set",
            "source_files": {
                TRAINING_CHALLENGES: _sha256(challenge_path),
                TRAINING_SOLUTIONS: _sha256(solutions_path),
            },
            "seed": seed,
            "original_public_training_tasks": len(challenges),
            "training_original_tasks": len(train_ids),
            "validation_original_tasks": len(validation_ids),
            "training_task_ids": train_ids,
            "validation_task_ids": validation_ids,
            "augmentation": {
                "requested_unique_variants_per_training_task": num_aug,
                "original_training_variants": len(train_ids),
                "generated_training_augmented_variants": len(train_puzzle_identifiers) - len(train_ids),
                "total_training_variants_including_original": len(train_puzzle_identifiers),
                "actual_variants_per_training_task": train_task_variant_counts,
            "original_grid_pairs_per_training_task": train_grid_pairs_per_task,
                "transformations": [
                    "8 dihedral rotations/reflections",
                    "permutation of colors 1..9 with color 0 fixed",
                    "random top/left translation of training pairs with one pair untranslated",
                ],
                "duplicate_variants": "deduplicated by the upstream task hash; symmetric tasks may yield fewer than requested",
            },
            "grid_pair_counts": {
                "unaugmented_training_task_pairs": training_original_pairs,
                "stored_augmented_training_grid_pairs": train_pair_total,
                "validation_original_grid_pairs": validation_pair_total,
            },
            "sampler": {
                "implementation": "upstream PuzzleDataset._iter_train / _sample_batch",
                "policy": "shuffle original task groups once per epoch; sample one stored variant per group; pack that variant's grid pairs; drop a final partial global batch",
                "examples_per_epoch": "logged from emitted global batches in each training epoch because packing and final partial-batch dropping affect the exact count",
            },
            "validation_protocol": {
                "task_level_holdout": True,
                "validation_variants_per_task": 1,
                "validation_variant_count": len(test_puzzle_identifiers),
                "validation_uses_only_public_training_task_test_pairs": True,
                "official_evaluation_or_private_test_data_used": False,
            },
        }
        with (temporary / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)

        if output_dir.exists():
            output_dir.rmdir()  # It was checked empty before staging.
        os.replace(temporary, output_dir)
        return manifest
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-tasks", type=int, default=100)
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=1000,
        help="Expected number of public training tasks; use --expected-task-count 2 for a tiny fixture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.source_dir,
        args.output_dir,
        num_aug=args.num_aug,
        seed=args.seed,
        validation_tasks=args.validation_tasks,
        expected_task_count=args.expected_task_count,
    )
    print(json.dumps({
        key: manifest[key]
        for key in (
            "original_public_training_tasks",
            "training_original_tasks",
            "validation_original_tasks",
            "augmentation",
            "grid_pair_counts",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
