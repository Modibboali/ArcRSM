from __future__ import annotations

import json

import numpy as np

from scripts.build_arc_agi2_from_kaggle import build_dataset


def _task(test_value: int) -> dict:
    return {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[test_value]], "output": [[test_value]]}],
    }


def test_kaggle_filename_adapter_delegates_to_upstream_builder(tmp_path) -> None:
    source = tmp_path / "competition"
    output = tmp_path / "built"
    source.mkdir()

    files = {
        "arc-agi_training_challenges.json": {"train-task": _task(2)},
        "arc-agi_training_solutions.json": {"train-task": [[[2]]]},
        "arc-agi_evaluation_challenges.json": {"eval-task": _task(3)},
        "arc-agi_evaluation_solutions.json": {"eval-task": [[[3]]]},
    }
    for name, value in files.items():
        (source / name).write_text(json.dumps(value), encoding="utf-8")

    build_dataset(source, output, num_aug=0, seed=7)

    metadata = json.loads((output / "test" / "dataset.json").read_text(encoding="utf-8"))
    assert metadata["seq_len"] == 900
    assert metadata["vocab_size"] == 12
    assert metadata["total_puzzles"] == 1
    test_inputs = np.load(output / "test" / "all__inputs.npy")
    assert test_inputs.shape == (1, 900)
    assert int(test_inputs[0, 0]) == 5  # ARC color 3 shifted by PAD/EOS tokens.
