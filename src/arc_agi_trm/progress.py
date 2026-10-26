"""Compact rank-zero tqdm progress display shared by all training architectures."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Any, Mapping


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_number(value: float | None, format_spec: str) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return format(value, format_spec).replace("e-0", "e-").replace("e+0", "e+")


class TrainingProgress:
    """One compact progress bar whose unit is a completed optimizer update."""

    def __init__(
        self,
        *,
        total_steps: int,
        initial_step: int,
        total_epochs: int,
        initial_epoch: int = 0,
        tqdm_factory=None,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("Training progress total_steps must be positive")
        if tqdm_factory is None:
            try:
                from tqdm.auto import tqdm as tqdm_factory
            except ImportError as exc:
                raise RuntimeError(
                    "Training progress requires tqdm. Install the project's training "
                    "requirements from requirements/kaggle-training.txt."
                ) from exc

        self._tqdm = tqdm_factory
        self.total_epochs = int(total_epochs)
        self._epoch = int(initial_epoch)
        self._start_time = time.monotonic()
        self._last_step_time = self._start_time
        self._step_times: deque[float] = deque(maxlen=30)
        self._example_samples: deque[tuple[float, int]] = deque(maxlen=31)
        self._example_samples.append((self._start_time, 0))
        self._examples = 0
        self._rolling_training: deque[tuple[float, int, float, float, float]] = deque(
            maxlen=30
        )
        self._last_learning_rate: float | None = None
        self._last_validation_loss: float | None = None
        self._last_validation_exact: float | None = None
        self._closed = False
        resolved_total = max(int(total_steps), int(initial_step))
        self._bar = tqdm_factory(
            total=resolved_total,
            initial=int(initial_step),
            desc=self._description(initial_step, resolved_total),
            unit="",
            position=0,
            leave=True,
            mininterval=0.5,
            smoothing=0.1,
            dynamic_ncols=True,
            bar_format="{desc} |{bar:10}| {postfix}",
        )
        self._render_postfix(refresh=False)

    def _set_description(self) -> None:
        description = self._description(self._bar.n, self._bar.total)
        set_description = getattr(self._bar, "set_description_str", None)
        if set_description is None:
            self._bar.set_description(description)
        else:
            set_description(description)

    def _description(self, step: int, total_steps: int) -> str:
        return (
            f"TRM | Epoch {self._epoch}/{self.total_epochs} | "
            f"Steps {int(step)}/{int(total_steps)}"
        )

    def _training_window(self) -> tuple[float | None, float | None, float | None]:
        if not self._rolling_training:
            return None, None, None
        loss_sum = sum(item[0] for item in self._rolling_training)
        examples = sum(item[1] for item in self._rolling_training)
        grid_sum = sum(item[2] for item in self._rolling_training)
        exact_sum = sum(item[3] for item in self._rolling_training)
        metric_count = sum(item[4] for item in self._rolling_training)
        return (
            loss_sum / max(1, examples),
            grid_sum / max(1.0, metric_count),
            exact_sum / max(1.0, metric_count),
        )

    def _postfix(self) -> str:
        now = time.monotonic()
        elapsed = max(0.0, now - self._start_time)
        samples = list(self._example_samples)
        if len(samples) >= 2 and samples[-1][0] > samples[0][0]:
            examples_per_second = (samples[-1][1] - samples[0][1]) / (
                samples[-1][0] - samples[0][0]
            )
        elif self._examples:
            examples_per_second = self._examples / max(elapsed, 1e-9)
        else:
            examples_per_second = None

        average_step_seconds = (
            sum(self._step_times) / len(self._step_times) if self._step_times else None
        )
        remaining_steps = max(0, int(self._bar.total) - int(self._bar.n))
        eta = (
            "--:--:--"
            if average_step_seconds is None
            else _format_duration(remaining_steps * average_step_seconds)
        )
        loss, grid_accuracy, exact_accuracy = self._training_window()
        fields = [
            f"Loss {_format_number(loss, '.3f')}",
            f"Grid {_format_number(None if grid_accuracy is None else grid_accuracy * 100, '.1f')}%",
            f"Exact {_format_number(None if exact_accuracy is None else exact_accuracy * 100, '.1f')}%",
            f"LR {_format_number(self._last_learning_rate, '.3g')}",
            f"{_format_number(examples_per_second, '.1f')} ex/s",
            f"Elapsed {_format_duration(elapsed)}",
            f"ETA ~{eta}",
        ]
        if self._last_validation_loss is not None:
            fields.append(f"Val loss {_format_number(self._last_validation_loss, '.3f')}")
        if self._last_validation_exact is not None:
            fields.append(
                "Val exact "
                f"{_format_number(self._last_validation_exact * 100, '.1f')}%"
            )
        return " | ".join(fields)

    def _render_postfix(self, *, refresh: bool) -> None:
        self._bar.set_postfix_str(self._postfix(), refresh=refresh)

    def optimizer_step(self, step: Mapping[str, Any]) -> None:
        if self._closed:
            return
        now = time.monotonic()
        self._step_times.append(max(0.0, now - self._last_step_time))
        self._last_step_time = now
        examples = int(step["examples"])
        self._examples += examples
        self._example_samples.append((now, self._examples))
        metric_count = max(0, int(step.get("training_metric_count", examples)))
        self._rolling_training.append(
            (
                float(step["training_loss"]) * examples,
                examples,
                float(step.get("training_grid_accuracy", 0.0)) * metric_count,
                float(step.get("training_batch_exact_accuracy", 0.0)) * metric_count,
                float(metric_count),
            )
        )
        if step.get("learning_rate") is not None:
            self._last_learning_rate = float(step["learning_rate"])
        self._epoch = int(step["epoch"])
        self._bar.update(1)
        self._set_description()
        self._render_postfix(refresh=False)

    def validation_completed(self, metrics: Mapping[str, Any]) -> None:
        """Show only actual validation results, with their validation labels."""

        if self._closed:
            return
        validation_loss = metrics.get("validation_loss")
        validation_exact = metrics.get("validation_task_exact_match")
        self._last_validation_loss = (
            float(validation_loss) if validation_loss is not None else None
        )
        self._last_validation_exact = (
            float(validation_exact) if validation_exact is not None else None
        )
        self._render_postfix(refresh=True)

    def epoch_completed(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        if self._closed:
            return
        self._epoch = int(epoch)
        self._set_description()
        self._render_postfix(refresh=True)

    def set_total(self, total_steps: int) -> None:
        if not self._closed:
            self._bar.total = max(int(total_steps), int(self._bar.n))
            self._set_description()
            self._render_postfix(refresh=True)

    def write(self, message: str) -> None:
        if self._closed:
            print(message, flush=True)
        else:
            self._tqdm.write(message)

    def close(self, *, final_step: int, status: str | None = None) -> None:
        if self._closed:
            return
        if status in {"complete", "smoke-complete", "diagnostic-stop"}:
            self._bar.total = max(int(final_step), int(self._bar.n))
        self._set_description()
        self._render_postfix(refresh=True)
        self._bar.close()
        self._closed = True
