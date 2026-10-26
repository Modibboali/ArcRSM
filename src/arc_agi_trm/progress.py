"""Rank-zero optimizer-step progress display for training runs."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Mapping


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    if minutes:
        return f"{minutes:d}:{seconds:02d}"
    return f"{seconds}s"


class TrainingProgress:
    """One compact tqdm bar whose unit is a completed optimizer update."""

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
        self._bar = tqdm_factory(
            total=max(total_steps, initial_step),
            initial=initial_step,
            desc=f"Epoch {initial_epoch}/{total_epochs}",
            unit="opt",
            position=0,
            leave=True,
            mininterval=1.0,
            smoothing=0.1,
            dynamic_ncols=True,
            bar_format=(
                "{desc} {bar:28} {n_fmt}/{total_fmt} opt "
                "[{elapsed}]{postfix}"
            ),
        )
        self.total_epochs = int(total_epochs)
        self._start_time = time.monotonic()
        self._step_times: deque[float] = deque(maxlen=30)
        self._example_samples: deque[tuple[float, int]] = deque(maxlen=31)
        self._example_samples.append((self._start_time, 0))
        self._examples = 0
        self._last_step_time = self._start_time
        self._last_loss: float | None = None
        self._last_learning_rate: float | None = None
        self._closed = False
        self._bar.set_postfix(self._postfix(), refresh=False)

    def _postfix(
        self,
        *,
        loss: float | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, str]:
        now = time.monotonic()
        elapsed = max(now - self._start_time, 1e-9)
        samples = list(self._example_samples)
        if len(samples) >= 2 and samples[-1][0] > samples[0][0]:
            examples_per_second = (samples[-1][1] - samples[0][1]) / (
                samples[-1][0] - samples[0][0]
            )
        elif self._examples:
            examples_per_second = self._examples / elapsed
        else:
            examples_per_second = 0.0

        average_step_seconds = (
            sum(self._step_times) / len(self._step_times) if self._step_times else None
        )
        remaining_steps = max(0, int(self._bar.total or self._bar.n) - int(self._bar.n))
        eta = (
            "?"
            if average_step_seconds is None
            else _format_duration(remaining_steps * average_step_seconds)
        )
        result = {
            "loss": f"{loss:.4g}" if loss is not None and math.isfinite(loss) else "?",
            "lr": f"{learning_rate:.2g}"
            if learning_rate is not None and math.isfinite(learning_rate)
            else "?",
            "ex/s": f"{examples_per_second:.1f}" if examples_per_second else "?",
            "eta~": eta,
        }
        return result

    def optimizer_step(self, step: Mapping[str, Any]) -> None:
        if self._closed:
            return
        now = time.monotonic()
        self._step_times.append(max(0.0, now - self._last_step_time))
        self._last_step_time = now
        self._examples += int(step["examples"])
        self._example_samples.append((now, self._examples))
        self._last_loss = float(step["training_loss"])
        self._last_learning_rate = (
            float(step["learning_rate"])
            if step.get("learning_rate") is not None
            else None
        )
        self._bar.set_description(f"Epoch {int(step['epoch'])}/{self.total_epochs}")
        self._bar.update(1)
        self._bar.set_postfix(
            self._postfix(
                loss=self._last_loss,
                learning_rate=self._last_learning_rate,
            ),
            refresh=False,
        )

    def epoch_completed(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        if self._closed:
            return
        self._bar.set_description(f"Epoch {epoch}/{self.total_epochs}")
        self._bar.set_postfix(
            self._postfix(loss=self._last_loss, learning_rate=self._last_learning_rate),
            refresh=False,
        )

    def set_total(self, total_steps: int) -> None:
        if not self._closed:
            self._bar.total = max(int(total_steps), int(self._bar.n))
            self._bar.refresh()

    def write(self, message: str) -> None:
        if self._closed:
            print(message, flush=True)
        else:
            self._tqdm.write(message)

    def close(self, *, final_step: int, status: str | None = None) -> None:
        if self._closed:
            return
        self._bar.total = max(0, int(final_step))
        self._bar.refresh()
        self._bar.close()
        self._closed = True
