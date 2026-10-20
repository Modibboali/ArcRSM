from __future__ import annotations

import json

import numpy as np

from arc_agi_trm.experiment import batch_sizes
from arc_agi_trm.runner import _crop_tokens
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from scripts.build_arc_agi2_from_kaggle import build_dataset


def _task(offset: int) -> dict:
    return {
        "train": [
            {
                "input": [[0, 1, 2], [3, 4, 5]],
                "output": [[5, 4, 3], [2, 1, 0]],
            }
        ],
        "test": [
            {
                "input": [[offset, 1], [2, 3]],
            }
        ],
    }


def _fixture(source, task_count: int = 4) -> None:
    source.mkdir()
    challenges = {f"task-{i}": _task(i) for i in range(task_count)}
    solutions = {task_id: [[[4, 3], [2, 1]]] for task_id in challenges}
    (source / "arc-agi_training_challenges.json").write_text(
        json.dumps(challenges), encoding="utf-8"
    )
    (source / "arc-agi_training_solutions.json").write_text(
        json.dumps(solutions), encoding="utf-8"
    )


def _sampler_signature(dataset_dir) -> list[tuple[int, ...]]:
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=19,
            dataset_paths=[str(dataset_dir)],
            global_batch_size=2,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="train",
    )
    np.random.seed(101)
    return [
        tuple(batch["puzzle_identifiers"].tolist())
        for _set_name, batch, _global_size in dataset
    ]


def test_streamed_builder_uses_training_tasks_only_and_is_reproducible(tmp_path) -> None:
    source = tmp_path / "competition"
    _fixture(source)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_dataset(
        source,
        first,
        num_aug=4,
        seed=7,
        validation_tasks=1,
        expected_task_count=4,
    )
    second_manifest = build_dataset(
        source,
        second,
        num_aug=4,
        seed=7,
        validation_tasks=1,
        expected_task_count=4,
    )

    train_ids = set(first_manifest["training_task_ids"])
    validation_ids = set(first_manifest["validation_task_ids"])
    assert len(train_ids) == 3
    assert len(validation_ids) == 1
    assert train_ids.isdisjoint(validation_ids)
    assert first_manifest["augmentation"]["requested_unique_variants_per_training_task"] == 4
    assert first_manifest["augmentation"]["original_training_variants"] == 3
    assert all(
        1 <= count <= 5
        for count in first_manifest["augmentation"]["actual_variants_per_training_task"].values()
    )
    assert first_manifest["validation_protocol"]["official_evaluation_or_private_test_data_used"] is False

    identifiers = json.loads((first / "identifiers.json").read_text(encoding="utf-8"))
    validation_puzzles = json.loads((first / "test_puzzles.json").read_text(encoding="utf-8"))
    assert set(validation_puzzles) == validation_ids
    assert set(name.split("|||")[0] for name in identifiers[1:]) == train_ids | validation_ids
    assert identifiers == json.loads((second / "identifiers.json").read_text(encoding="utf-8"))
    assert first_manifest["validation_task_ids"] == second_manifest["validation_task_ids"]
    assert _sampler_signature(first) == _sampler_signature(second)

    for split in ("train", "test"):
        for name in ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"):
            left = np.load(first / split / f"all__{name}.npy")
            right = np.load(second / split / f"all__{name}.npy")
            assert np.array_equal(left, right)
    train_inputs = np.load(first / "train" / "all__inputs.npy")
    train_indices = np.load(first / "train" / "all__puzzle_indices.npy")
    train_groups = np.load(first / "train" / "all__group_indices.npy")
    assert train_inputs.shape == (first_manifest["grid_pair_counts"]["stored_augmented_training_grid_pairs"], 900)
    assert train_indices[-1] == len(train_inputs)
    assert train_groups[-1] == len(train_indices) - 1


def test_batch_plan_reports_per_device_and_accumulated_global_sizes() -> None:
    assert batch_sizes(32, 4, 2) == {
        "per_device_batch_size": 8,
        "microbatch_global_size": 32,
        "gradient_accumulation_steps": 2,
        "effective_global_batch_size": 64,
    }


def test_crop_tokens_matches_arc_30_by_30_format() -> None:
    tokens = np.zeros(900, dtype=np.int32)
    tokens[:2] = [2, 3]
    tokens[30:32] = [4, 5]
    assert _crop_tokens(tokens).tolist() == [[0, 1], [2, 3]]
