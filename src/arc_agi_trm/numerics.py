"""Opt-in finite-value diagnostics for training runs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist


def _tensor_count(value: torch.Tensor, mask: torch.Tensor) -> int:
    return int(mask.sum().detach().cpu().item())


class NumericsDiagnostics:
    """Collect the first bad value in a batch and coordinate the failure by rank."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        cpu_group=None,
        log_path: str | Path | None = None,
        puzzle_names: list[str] | None = None,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.cpu_group = cpu_group
        self.log_path = Path(log_path) if log_path is not None else None
        self.puzzle_names = puzzle_names
        self.epoch = 0
        self.optimizer_step = 0
        self.batch_ids: list[int] = []
        self.batch_names: list[str] = []
        self.first_failure: dict[str, Any] | None = None

    def set_context(
        self,
        *,
        epoch: int,
        optimizer_step: int,
        batch: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.epoch = int(epoch)
        self.optimizer_step = int(optimizer_step)
        if batch is not None and "puzzle_identifiers" in batch:
            raw_ids = batch["puzzle_identifiers"].detach().reshape(-1).cpu().tolist()
            self.batch_ids = sorted({int(item) for item in raw_ids})
            self.batch_names = (
                [
                    self.puzzle_names[item]
                    if 0 <= item < len(self.puzzle_names)
                    else f"<unknown:{item}>"
                    for item in self.batch_ids
                ]
                if self.puzzle_names is not None
                else []
            )

    def _base_record(self, *, stage: str, tensor_name: str) -> dict[str, Any]:
        return {
            "event": "numerical_failure",
            "rank": self.rank,
            "epoch": self.epoch,
            "optimizer_step": self.optimizer_step,
            "stage": stage,
            "tensor_name": tensor_name,
            "puzzle_identifiers": self.batch_ids,
            "puzzle_names": self.batch_names,
        }

    def _record(self, record: dict[str, Any]) -> None:
        if self.first_failure is None:
            self.first_failure = record

    def check_tensor(self, name: str, value: torch.Tensor, *, stage: str) -> None:
        if self.first_failure is not None or not isinstance(value, torch.Tensor):
            return
        if value.is_floating_point() or value.is_complex():
            finite = torch.isfinite(value)
            count = _tensor_count(value, ~finite)
            if count:
                record = self._base_record(stage=stage, tensor_name=name)
                record.update(
                    failure_kind="non_finite",
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    non_finite_count=count,
                )
                self._record(record)

    def check_range(
        self,
        name: str,
        value: torch.Tensor,
        *,
        stage: str,
        minimum: int,
        maximum: int,
        ignore_value: int | None = None,
    ) -> None:
        if self.first_failure is not None:
            return
        valid = (value >= minimum) & (value <= maximum)
        if ignore_value is not None:
            valid |= value == ignore_value
        invalid_count = _tensor_count(value, ~valid)
        if invalid_count:
            record = self._base_record(stage=stage, tensor_name=name)
            record.update(
                failure_kind="out_of_range",
                shape=list(value.shape),
                dtype=str(value.dtype),
                non_finite_count=0,
                invalid_count=invalid_count,
                valid_range=[minimum, maximum],
                allowed_ignore_value=ignore_value,
            )
            self._record(record)

    def check_scalar(self, name: str, value: float, *, stage: str) -> None:
        if self.first_failure is None and not math.isfinite(float(value)):
            record = self._base_record(stage=stage, tensor_name=name)
            record.update(
                failure_kind="non_finite",
                shape=[],
                dtype="python.float",
                non_finite_count=1,
            )
            self._record(record)

    def check_tree(self, name: str, value: Any, *, stage: str) -> None:
        if self.first_failure is not None:
            return
        if isinstance(value, torch.Tensor):
            self.check_tensor(name, value, stage=stage)
        elif isinstance(value, float) and not math.isfinite(value):
            record = self._base_record(stage=stage, tensor_name=name)
            record.update(
                failure_kind="non_finite",
                shape=[],
                dtype="python.float",
                non_finite_count=1,
            )
            self._record(record)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                self.check_tree(f"{name}.{key}", item, stage=stage)
                if self.first_failure is not None:
                    return
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.check_tree(f"{name}[{index}]", item, stage=stage)
                if self.first_failure is not None:
                    return
        elif hasattr(value, "__dataclass_fields__"):
            for field_name in value.__dataclass_fields__:
                self.check_tree(f"{name}.{field_name}", getattr(value, field_name), stage=stage)
                if self.first_failure is not None:
                    return

    def check_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        vocab_size: int,
        num_puzzle_identifiers: int,
    ) -> None:
        for name, tensor in batch.items():
            self.check_tensor(f"batch.{name}", tensor, stage="batch")
            if self.first_failure is not None:
                return
        if "inputs" in batch:
            self.check_range(
                "batch.inputs",
                batch["inputs"],
                stage="batch",
                minimum=0,
                maximum=vocab_size - 1,
            )
        if self.first_failure is None and "labels" in batch:
            self.check_range(
                "batch.labels",
                batch["labels"],
                stage="batch",
                minimum=0,
                maximum=vocab_size - 1,
                ignore_value=-100,
            )
        if self.first_failure is None and "puzzle_identifiers" in batch:
            self.check_range(
                "batch.puzzle_identifiers",
                batch["puzzle_identifiers"],
                stage="batch",
                minimum=0,
                maximum=num_puzzle_identifiers - 1,
            )

    def _emit(self, record: dict[str, Any], *, reporting_rank: int) -> None:
        rendered = dict(record)
        rendered["reporting_rank"] = reporting_rank
        line = json.dumps(rendered, sort_keys=True, allow_nan=False)
        print(line, file=sys.stderr, flush=True)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()

    def synchronize(self, stage: str, *, device: torch.device) -> None:
        """Raise the same source failure on every rank at a safe collective boundary."""

        local_bad = self.first_failure is not None
        if self.world_size > 1:
            flag = torch.tensor([int(local_bad)], dtype=torch.int32, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if not bool(flag.item()):
                return
            gathered: list[dict[str, Any] | None] = [None] * self.world_size
            dist.all_gather_object(gathered, self.first_failure, group=self.cpu_group)
            failures = [item for item in gathered if item is not None]
        elif local_bad:
            failures = [self.first_failure]
        else:
            return

        for item in failures:
            self._emit(item, reporting_rank=self.rank)
        details = "; ".join(
            f"rank={item['rank']} epoch={item['epoch']} step={item['optimizer_step']} "
            f"tensor={item['tensor_name']} kind={item['failure_kind']} "
            f"count={item.get('non_finite_count', item.get('invalid_count', 0))}"
            for item in failures
        )
        raise FloatingPointError(
            f"Numerics diagnostic found the first non-finite/invalid value at stage={stage}: "
            f"{details}"
        )
