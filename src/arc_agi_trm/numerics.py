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
        detailed_step_range: tuple[int, int] | None = None,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.cpu_group = cpu_group
        self.log_path = Path(log_path) if log_path is not None else None
        self.puzzle_names = puzzle_names
        if detailed_step_range is not None:
            first_step, last_step = (int(value) for value in detailed_step_range)
            if first_step < 1 or last_step < first_step:
                raise ValueError("detailed_step_range must be a positive inclusive range")
            self.detailed_step_range = (first_step, last_step)
        else:
            self.detailed_step_range = None
        self.training_log_path = (
            self.log_path.with_name(f"training_diagnostics_rank{self.rank}.jsonl")
            if self.log_path is not None
            else None
        )
        self.epoch = 0
        self.optimizer_step = 0
        self.batch_ids: list[int] = []
        self.batch_names: list[str] = []
        self.first_failure: dict[str, Any] | None = None
        self.propagation_log_path = (
            self.log_path.with_name(
                "propagation_" + self.log_path.name.removeprefix("numerics_")
            )
            if self.log_path is not None
            else None
        )
        self.forward_id = 0
        self.l_level_call_index = 0
        self.recursion_shape: tuple[int, int, int] | None = None
        self.current_recursion_call: dict[str, Any] | None = None
        self.invocation_by_module: dict[str, int] = {}
        self.invocations: list[dict[str, Any]] = []
        self.q_head_forward_call_index = 0

    @staticmethod
    def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
        """Return compact activation/gradient statistics without retaining tensors."""

        source = value.detach()
        if source.is_complex():
            source = source.abs()
        source = source.to(torch.float32)
        finite = torch.isfinite(source)
        total_count = int(source.numel())
        finite_count_tensor = finite.sum()
        result: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "element_count": total_count,
        }
        safe = torch.where(finite, source, 0.0)
        max_abs_tensor = safe.abs().max()
        scale = max_abs_tensor.clamp_min(torch.finfo(source.dtype).tiny)
        scaled = safe / scale
        count = finite_count_tensor.clamp_min(1).to(torch.float32)
        scaled_sum = scaled.sum()
        scaled_square_sum = scaled.square().sum()
        scaled_mean = scaled_sum / count
        centered = torch.where(finite, scaled - scaled_mean, 0.0)
        scaled_variance = centered.square().sum() / count
        minimum_tensor = torch.where(finite, source, float("inf")).min()
        maximum_tensor = torch.where(finite, source, float("-inf")).max()
        packed = torch.stack((
            finite_count_tensor.to(torch.float32),
            max_abs_tensor,
            torch.sqrt(scaled_square_sum / count),
            torch.sqrt(scaled_variance),
            torch.sqrt(scaled_square_sum),
            minimum_tensor,
            maximum_tensor,
        )).cpu().tolist()
        finite_count = int(packed[0])
        result.update(
            finite_count=finite_count,
            non_finite_count=total_count - finite_count,
        )
        if not finite_count:
            result.update(rms=None, std=None, l2_norm=None, max_abs=None, minimum=None, maximum=None)
            return result
        max_abs = float(packed[1])
        result.update(
            rms=max_abs * float(packed[2]),
            std=max_abs * float(packed[3]),
            l2_norm=max_abs * float(packed[4]),
            max_abs=max_abs,
            minimum=float(packed[5]),
            maximum=float(packed[6]),
        )
        return result

    def _append_propagation_record(self, record: dict[str, Any]) -> None:
        rendered = {
            "rank": self.rank,
            "epoch": self.epoch,
            "optimizer_step": self.optimizer_step,
            **record,
        }
        line = json.dumps(rendered, sort_keys=True, allow_nan=False)
        if self.propagation_log_path is not None:
            self.propagation_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.propagation_log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()

    def _append_training_record(self, record: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        rendered = {
            "rank": self.rank,
            "epoch": self.epoch,
            "optimizer_step": self.optimizer_step,
            **record,
        }
        line = json.dumps(rendered, sort_keys=True, allow_nan=False)
        if self.training_log_path is not None:
            self.training_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.training_log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()

    def is_detailed_step(self) -> bool:
        return (
            self.detailed_step_range is None
            or self.detailed_step_range[0] <= self.optimizer_step <= self.detailed_step_range[1]
        )

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().item()
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _small_tensor_values(value: torch.Tensor, *, limit: int = 256) -> list[Any] | None:
        if value.numel() > limit:
            return None
        values = value.detach().cpu().reshape(-1).tolist()

        def safe(item: Any) -> Any:
            if isinstance(item, list):
                return [safe(nested) for nested in item]
            if isinstance(item, float) and not math.isfinite(item):
                return None
            return item

        return safe(values)

    def record_q_head_activation(
        self, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Capture the Q-head input/output and their backward gradients in the step window."""

        if self.rank != 0 or not self.is_detailed_step():
            return
        call_index = self.q_head_forward_call_index
        self.q_head_forward_call_index += 1
        input_tensor = inputs[0] if inputs else None
        self._append_training_record({
            "event": "q_head_forward",
            "call_index": call_index,
            "input": self.tensor_stats(input_tensor) if isinstance(input_tensor, torch.Tensor) else None,
            "output": self.tensor_stats(output),
            "output_values": self._small_tensor_values(output),
        })
        for tensor_name, tensor in (("input", input_tensor), ("output", output)):
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(
                    lambda gradient, name=tensor_name, call=call_index: (
                        self._append_training_record({
                            "event": "q_head_activation_gradient",
                            "call_index": call,
                            "tensor": name,
                            "stats": self.tensor_stats(gradient),
                        }),
                        gradient,
                    )[1]
                )

    def record_loss_batch(
        self,
        *,
        loss: torch.Tensor,
        metrics: Mapping[str, Any],
        outputs: Mapping[str, torch.Tensor] | None,
        batch: Mapping[str, torch.Tensor],
        model: torch.nn.Module,
    ) -> None:
        """Write raw ACT loss, components, masks, Q values and logit ranges."""

        if self.rank != 0 or not self.is_detailed_step():
            return
        outputs = outputs or {}
        loss_components = {
            name: {
                "shape": list(value.shape) if isinstance(value, torch.Tensor) else [],
                "stats": self.tensor_stats(value) if isinstance(value, torch.Tensor) else None,
                "value": self._safe_float(value),
            }
            for name, value in metrics.items()
            if name.endswith("_loss")
        }
        metric_values = {
            name: self._safe_float(value)
            for name, value in metrics.items()
            if not name.endswith("_loss")
        }
        loss_semantics = {
            "total_formula": "lm_loss + 0.5 * (q_halt_loss + q_continue_loss)",
            "lm_loss": {
                "token_loss": "stablemax_cross_entropy; ignore_index=-100; valid_mask=labels != -100",
                "sequence_denominator": "valid token count per row, clamped to 1",
                "reduction": "sum(token_loss / per_row_denominator) over batch and sequence",
            },
            "q_halt_loss": {
                "function": "binary_cross_entropy_with_logits",
                "target": "sequence_correct: every non-ignored label token predicted correctly",
                "reduction": "sum over batch; no halted-example mask",
            },
            "q_continue_loss": {
                "function": "binary_cross_entropy_with_logits",
                "target": "detached target_q_continue when ACT continuation is enabled",
                "reduction": "sum over batch; absent when target_q_continue is absent",
            },
        }
        labels = batch.get("labels")
        valid_mask = labels.ne(-100) if isinstance(labels, torch.Tensor) else None
        per_row_valid_counts = (
            valid_mask.sum(dim=-1) if valid_mask is not None else None
        )
        sequence_correct = None
        if (
            valid_mask is not None
            and isinstance(outputs.get("preds"), torch.Tensor)
        ):
            preds = outputs["preds"]
            token_correct = valid_mask & preds.eq(labels)
            sequence_correct = token_correct.sum(dim=-1).eq(valid_mask.sum(dim=-1))

        q_head_parameters = {}
        for name, parameter in model.named_parameters():
            if name.endswith(("inner.q_head.weight", "inner.q_head.bias")):
                q_head_parameters[name] = self.tensor_stats(parameter)
        prediction_logits = outputs.get("logits")
        tensor_stats = {
            name: self.tensor_stats(value)
            for name in ("q_halt_logits", "q_continue_logits", "target_q_continue")
            if isinstance((value := outputs.get(name)), torch.Tensor)
        }
        q_values = {
            name: self._small_tensor_values(value)
            for name in ("q_halt_logits", "q_continue_logits", "target_q_continue")
            if isinstance((value := outputs.get(name)), torch.Tensor)
        }
        self._append_training_record({
            "event": "loss_batch_before_backward",
            "raw_backward_loss": {
                "shape": list(loss.shape),
                "stats": self.tensor_stats(loss),
                "value": self._safe_float(loss),
            },
            "loss_components": loss_components,
            "loss_component_shapes": {
                name: record["shape"] for name, record in loss_components.items()
            },
            "loss_semantics": loss_semantics,
            "batch_reductions": {
                "labels_shape": list(labels.shape) if isinstance(labels, torch.Tensor) else None,
                "valid_token_mask_shape": list(valid_mask.shape) if valid_mask is not None else None,
                "valid_tokens_per_row": (
                    per_row_valid_counts.detach().cpu().tolist()
                    if per_row_valid_counts is not None else None
                ),
                "lm_denominators_after_clamp_min_1": (
                    per_row_valid_counts.clamp_min(1).detach().cpu().tolist()
                    if per_row_valid_counts is not None else None
                ),
                "q_halt_targets_sequence_correct": (
                    sequence_correct.detach().cpu().tolist()
                    if sequence_correct is not None else None
                ),
                "halt_loss_mask": None,
                "q_losses_divided_by_batch_size": False,
            },
            "outputs": {
                "prediction_logits": (
                    self.tensor_stats(prediction_logits)
                    if isinstance(prediction_logits, torch.Tensor) else None
                ),
                "q_tensors": tensor_stats,
                "q_values": q_values,
            },
            "q_head_parameters_before_backward": q_head_parameters,
            "other_metrics": metric_values,
        })

    def record_adam_atan2_square_overflow_risk(
        self, optimizer: torch.optim.Optimizer, model: torch.nn.Module
    ) -> None:
        """Stop before AdamATan2's fused `grad * grad` overflows its opmath dtype."""

        if optimizer.__class__.__name__ != "AdamATan2":
            return
        for _name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or not gradient.is_floating_point():
                continue
            opmath_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
            square_limit = math.sqrt(torch.finfo(opmath_dtype).max)
            absolute_gradient = gradient.detach().to(opmath_dtype).abs()
            overflow_mask = absolute_gradient > square_limit
            overflow_count = int(overflow_mask.sum().detach().cpu().item())
            if overflow_count:
                stats = self.tensor_stats(gradient)
                self._record({
                    **self._base_record(
                        stage="before_optimizer_step",
                        tensor_name=f"gradient.{_name}.squared_for_exp_avg_sq",
                    ),
                    "failure_kind": "overflow_if_squared",
                    "shape": list(gradient.shape),
                    "dtype": str(gradient.dtype),
                    "non_finite_count": 0,
                    "overflow_count": overflow_count,
                    "max_abs_gradient": stats["max_abs"],
                    "safe_max_abs_gradient": square_limit,
                    "opmath_dtype": str(opmath_dtype),
                    "first_invalid_operation": (
                        "AdamATan2 CUDA kernel computes grad * grad before updating exp_avg_sq"
                    ),
                })
                return

    def begin_recursion_forward(
        self, *, h_cycles: int, l_cycles: int, l_layers: int
    ) -> None:
        """Start labels for one upstream Inner.forward recursion."""

        self.forward_id += 1
        self.l_level_call_index = 0
        self.recursion_shape = (int(h_cycles), int(l_cycles), int(l_layers))
        self.current_recursion_call = None
        self.invocation_by_module = {}
        self.invocations = []

    def begin_l_level_call(self) -> None:
        if self.recursion_shape is None:
            return
        h_cycles, l_cycles, _l_layers = self.recursion_shape
        call_index = self.l_level_call_index
        self.l_level_call_index += 1
        self.invocation_by_module = {}
        if call_index >= h_cycles * (l_cycles + 1):
            self.current_recursion_call = None
            return
        calls_per_h_cycle = l_cycles + 1
        h_cycle = min(call_index // calls_per_h_cycle, h_cycles - 1)
        update_index = call_index % calls_per_h_cycle
        is_l_update = update_index < l_cycles
        self.current_recursion_call = {
            "forward_id": self.forward_id,
            "l_level_call_index": call_index,
            "h_cycle": h_cycle,
            "latent_update": f"L{update_index}" if is_l_update else "H",
            "autograd_expected": h_cycle == h_cycles - 1,
        }

    def start_block_invocation(
        self, module_path: str, layer_index: int | None, hidden_states: torch.Tensor
    ) -> int | None:
        if self.current_recursion_call is None or not self.is_detailed_step():
            return None
        invocation_id = len(self.invocations)
        label = dict(self.current_recursion_call)
        label.update(
            event="recursion_invocation",
            invocation_id=invocation_id,
            module=module_path,
            layer_index=layer_index,
            autograd_enabled=bool(torch.is_grad_enabled()),
        )
        label["forward"] = {"block_input": self.tensor_stats(hidden_states)}
        label["gradient"] = {}
        self.invocations.append(label)
        self.invocation_by_module[module_path] = invocation_id
        self._attach_invocation_gradient(invocation_id, "block_input", hidden_states)
        return invocation_id

    def record_invocation_tensor(
        self, module_path: str, operation: str, value: torch.Tensor
    ) -> None:
        if not self.is_detailed_step():
            return
        invocation_id = self.invocation_by_module.get(module_path)
        if invocation_id is None:
            return
        event = self.invocations[invocation_id]
        event["forward"][operation] = self.tensor_stats(value)
        self._attach_invocation_gradient(invocation_id, operation, value)

    def _attach_invocation_gradient(
        self, invocation_id: int, tensor_name: str, value: torch.Tensor
    ) -> None:
        if value.requires_grad and torch.is_grad_enabled():
            value.register_hook(
                lambda gradient, invocation_id=invocation_id, tensor_name=tensor_name:
                    self._record_invocation_gradient(invocation_id, tensor_name, gradient)
            )

    def _record_invocation_gradient(
        self, invocation_id: int, tensor_name: str, gradient: torch.Tensor
    ) -> torch.Tensor:
        self.invocations[invocation_id]["gradient"][tensor_name] = self.tensor_stats(gradient)
        return gradient

    def end_block_invocation(self, module_path: str, output: torch.Tensor) -> None:
        self.record_invocation_tensor(module_path, "block_output", output)

    @staticmethod
    def _trend(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "direction": "unavailable"}
        start, end = values[0], values[-1]
        ratio = (end / start) if start != 0 else None
        if ratio is not None and not math.isfinite(ratio):
            ratio = None
        increasing_steps = sum(right > left for left, right in zip(values, values[1:]))
        decreasing_steps = sum(right < left for left, right in zip(values, values[1:]))
        direction = "flat"
        if end > start and increasing_steps >= decreasing_steps:
            direction = "increasing"
        elif end < start and decreasing_steps >= increasing_steps:
            direction = "decreasing"
        elif increasing_steps or decreasing_steps:
            direction = "mixed"
        return {
            "count": len(values),
            "start": start,
            "end": end,
            "end_to_start_ratio": ratio,
            "direction": direction,
            "increasing_steps": increasing_steps,
            "decreasing_steps": decreasing_steps,
        }

    def finish_recursion_forward(self) -> None:
        """Emit forward/gradient summaries and recurrent norm trends for this batch."""

        if not self.invocations or not self.is_detailed_step():
            self.invocations = []
            return
        activation_groups: dict[str, dict[str, list[float]]] = {}
        gradient_groups: dict[str, dict[str, list[float]]] = {}
        for event in self.invocations:
            group = (
                f"H{event['h_cycle']}.layer{event['layer_index']}"
            )
            activation_groups.setdefault(group, {})
            gradient_groups.setdefault(group, {})
            for tensor_name in ("block_input", "block_output", "mamba_residual", "mlp_residual"):
                stats = event["forward"].get(tensor_name)
                if stats and stats.get("rms") is not None:
                    activation_groups[group].setdefault(tensor_name, []).append(stats["rms"])
                grad_stats = event["gradient"].get(tensor_name)
                if grad_stats and grad_stats.get("l2_norm") is not None:
                    gradient_groups[group].setdefault(tensor_name, []).append(grad_stats["l2_norm"])
        self._append_propagation_record(
            {
                "event": "recursion_forward_summary",
                "forward_id": self.forward_id,
                "invocations": self.invocations,
                "activation_trends": {
                    group: {name: self._trend(values) for name, values in stages.items()}
                    for group, stages in activation_groups.items()
                },
                "gradient_trends": {
                    group: {name: self._trend(values) for name, values in stages.items()}
                    for group, stages in gradient_groups.items()
                },
            }
        )
        self.invocations = []

    def record_model_gradients(
        self,
        model: torch.nn.Module,
        optimizers: tuple[torch.optim.Optimizer, ...] | list[torch.optim.Optimizer] = (),
        *,
        phase: str = "after_distributed_gradient_reduction",
    ) -> None:
        """Record per-parameter, module, and whole-model norms at a gradient stage."""

        if not self.is_detailed_step():
            return
        parameters = [(name, parameter) for name, parameter in model.named_parameters()]
        parameter_records: dict[str, Any] = {}
        module_squared_norms: dict[str, float] = {}
        module_max_abs: dict[str, float] = {}
        model_squared_norm = 0.0
        model_max_abs = 0.0
        for name, parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            stats = self.tensor_stats(gradient)
            parameter_records[name] = stats
            norm = stats["l2_norm"] or 0.0
            model_squared_norm += norm * norm
            model_max_abs = max(model_max_abs, stats["max_abs"] or 0.0)
            parts = name.split(".")[:-1]
            for depth in range(len(parts) + 1):
                module_path = ".".join(parts[:depth]) or "<root>"
                module_squared_norms[module_path] = module_squared_norms.get(module_path, 0.0) + norm * norm
                module_max_abs[module_path] = max(module_max_abs.get(module_path, 0.0), stats["max_abs"] or 0.0)
        module_records = {
            name: {"l2_norm": math.sqrt(value), "max_abs": module_max_abs[name]}
            for name, value in module_squared_norms.items()
        }
        optimizer_group_records = []
        for optimizer_index, optimizer in enumerate(optimizers):
            for group_index, group in enumerate(optimizer.param_groups):
                params = [
                    parameter for parameter in group["params"]
                    if parameter.grad is not None
                ]
                norm = math.sqrt(sum(
                    float(torch.sum(parameter.grad.detach().to(torch.float64).square()).cpu().item())
                    for parameter in params
                )) if params else 0.0
                optimizer_group_records.append({
                    "optimizer_index": optimizer_index,
                    "optimizer_class": optimizer.__class__.__name__,
                    "group_index": group_index,
                    "parameter_count_with_gradient": len(params),
                    "l2_norm": norm,
                })
        q_head_name = next((name for name, _parameter in parameters if name.endswith("inner.q_head.weight")), None)
        self._append_propagation_record({
            "event": "gradient_summary",
            "phase": phase,
            "forward_id": self.forward_id,
            "total_model_gradient": {
                "l2_norm": math.sqrt(model_squared_norm),
                "max_abs": model_max_abs,
                "parameter_count_with_gradient": len(parameter_records),
            },
            "optimizer_parameter_groups": optimizer_group_records,
            "module_gradient_groups": module_records,
            "parameter_gradients": parameter_records,
            "q_head_weight_name": q_head_name,
            "q_head_weight_gradient": parameter_records.get(q_head_name) if q_head_name else None,
        })

    def record_q_head_optimizer_state(
        self, optimizer: torch.optim.Optimizer, model: torch.nn.Module, *, phase: str
    ) -> None:
        parameter = next(
            (parameter for name, parameter in model.named_parameters()
             if name.endswith("inner.q_head.weight")),
            None,
        )
        state = optimizer.state.get(parameter, {}) if parameter is not None else {}
        fields = {}
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            fields[key] = self.tensor_stats(value) if isinstance(value, torch.Tensor) else None
            if isinstance(value, torch.Tensor):
                self.check_tensor(
                    f"optimizer[1].state.model.inner.q_head.weight.{key}",
                    value,
                    stage=phase,
                )
        parameter_stats = (
            self.tensor_stats(parameter) if isinstance(parameter, torch.Tensor) else None
        )
        parameter_group = None
        if parameter is not None:
            for group_index, group in enumerate(optimizer.param_groups):
                if any(item is parameter for item in group["params"]):
                    parameter_group = {
                        "group_index": group_index,
                        "lr": self._safe_float(group.get("lr")),
                        "betas": list(group.get("betas", ())) or None,
                        "epsilon": self._safe_float(group.get("eps")),
                        "weight_decay": self._safe_float(group.get("weight_decay")),
                        "parameter_count": len(group["params"]),
                    }
                    break
        step = state.get("step")
        self._append_propagation_record({
            "event": "q_head_optimizer_state",
            "phase": phase,
            "optimizer_class": optimizer.__class__.__name__,
            "optimizer_module": optimizer.__class__.__module__,
            "parameter_found": parameter is not None,
            "parameter_stats": parameter_stats,
            "parameter_group": parameter_group,
            "state_initialized": bool(state),
            "step": self._safe_float(step),
            "fields": fields,
        })

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
            self.q_head_forward_call_index = 0
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
