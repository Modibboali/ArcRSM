"""Run manual CUDA/Mamba qualification in the actual Kaggle GPU notebook."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import platform
import sys
import traceback
from typing import Callable

import torch

from kaggle_trm_runtime import (
    assert_finite_gradients,
    build_loss_model,
    check_training_result,
    make_batch,
    make_config,
    memory_report,
    model_step,
    synchronize,
)


class Qualification:
    def __init__(self) -> None:
        self.sections: list[dict[str, object]] = []
        self.environment: dict[str, object] = {}
        self.memory: dict[str, object] = {}
        self.cuda_ok = False
        self.dependencies_ok = False
        self.required_failure = False

    def run(
        self,
        name: str,
        fn: Callable[[], object] | None,
        *,
        required: bool = True,
        skip_reason: str | None = None,
    ):
        if skip_reason is not None:
            item = {"section": name, "status": "SKIP", "reason": skip_reason, "required": required}
            self.sections.append(item)
            print(f"[SKIP] {name}: {skip_reason}")
            return None
        try:
            result = fn() if fn is not None else None
            item = {"section": name, "status": "PASS", "result": result, "required": required}
            self.sections.append(item)
            print(f"[PASS] {name}: {json.dumps(result, sort_keys=True, default=str)}")
            return result
        except BaseException as exc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            item = {
                "section": name,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "required": required,
            }
            self.sections.append(item)
            if required:
                self.required_failure = True
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stdout)
            return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=2)
    parser.add_argument("--l-cycles", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Also try torch.compile after eager tests")
    return parser.parse_args()


def environment_check(report: Qualification) -> dict[str, object]:
    triton_version = None
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        pass
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": devices,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "triton": triton_version,
    }
    report.environment.update(info)

    problems = []
    if platform.python_version() != "3.12.13":
        problems.append(f"Python is {platform.python_version()}, expected 3.12.13")
    if not str(torch.__version__).startswith("2.10.0+cu128"):
        problems.append(f"torch is {torch.__version__}, expected 2.10.0+cu128")
    if torch.version.cuda != "12.8":
        problems.append(f"PyTorch CUDA build is {torch.version.cuda}, expected 12.8")
    if not torch.cuda.is_available():
        problems.append("CUDA is not available")
    if torch.cuda.device_count() != 4:
        problems.append(f"found {torch.cuda.device_count()} GPUs, expected 4 L4 GPUs")
    if not info["cxx11_abi"]:
        problems.append("PyTorch CXX11 ABI is false, but the target wheels require true")
    if triton_version != "3.6.0":
        problems.append(f"Triton is {triton_version}, expected 3.6.0")
    if devices and not all("L4" in d["name"] and d["compute_capability"] == [8, 9] for d in devices):
        problems.append("one or more devices are not NVIDIA L4 (compute capability 8.9)")
    report.cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if problems:
        raise RuntimeError("; ".join(problems) + "\nObserved environment: " + json.dumps(info))
    return info


def dependency_check() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name, package_name in (
        ("mamba_ssm", "mamba-ssm"),
        ("causal_conv1d", "causal-conv1d"),
    ):
        module = importlib.import_module(module_name)
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = str(getattr(module, "__version__", "unknown"))
        versions[package_name] = version
    expected = {"mamba-ssm": "2.3.2.post1", "causal-conv1d": "1.6.2.post1"}
    mismatches = [
        f"{package_name} is {version}, expected {expected[package_name]}"
        for package_name, version in versions.items()
        if version != expected[package_name]
    ]
    if mismatches:
        raise RuntimeError("; ".join(mismatches))
    module = importlib.import_module("mamba_ssm.modules.mamba2")
    if not hasattr(module, "Mamba2"):
        raise RuntimeError("mamba_ssm.modules.mamba2 does not export Mamba2")
    return versions


def standalone_mamba2(batch_size: int, device: torch.device) -> dict[str, object]:
    from mamba_ssm.modules.mamba2 import Mamba2

    module = Mamba2(
        d_model=128,
        d_state=128,
        headdim=64,
        expand=2,
        device=device,
        dtype=torch.bfloat16,
    )
    values = torch.randn(batch_size, 900, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    output = module(values)
    if output.shape != values.shape or output.dtype != values.dtype or output.device != values.device:
        raise RuntimeError(
            f"Unexpected Mamba2 boundary: input={tuple(values.shape)}/{values.dtype}/{values.device}, "
            f"output={tuple(output.shape)}/{output.dtype}/{output.device}"
        )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Standalone Mamba2 output contains non-finite values")
    output.float().square().mean().backward()
    parameter_count = assert_finite_gradients(module)
    if values.grad is None or not bool(torch.isfinite(values.grad).all()):
        raise RuntimeError("Standalone Mamba2 input gradient is missing or non-finite")
    return {"shape": list(output.shape), "dtype": str(output.dtype), "trainable_parameters": parameter_count}


def trm_cuda_test(
    operator: str,
    config: dict[str, object],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    report: Qualification,
) -> dict[str, object]:
    loss_model = build_loss_model(config, operator=operator, device=device)
    loss_model.train()
    loss_model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    result = model_step(loss_model, batch)
    carry, loss, metrics, outputs = check_training_result(
        result,
        batch_size=batch["inputs"].shape[0],
        seq_len=batch["inputs"].shape[1],
    )
    loss.backward()
    parameter_count = assert_finite_gradients(loss_model)
    synchronize(device)
    memory = memory_report(device)
    report.memory[operator] = memory
    answer = {
        "operator": operator,
        "input_shape": list(batch["inputs"].shape),
        "carry_shape": list(carry.inner_carry.z_H.shape),
        "logits_shape": list(outputs["logits"].shape),
        "q_head_shape": list(outputs["q_halt_logits"].shape),
        "loss": float(loss.detach().float().cpu()),
        "lm_loss": float(metrics["lm_loss"].detach().float().cpu()),
        "trainable_parameters_with_gradients": parameter_count,
        **memory,
    }
    del loss_model, result, carry, outputs, metrics, loss
    gc.collect()
    torch.cuda.empty_cache()
    return answer


def compile_test(
    operator: str,
    config: dict[str, object],
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    loss_model = build_loss_model(config, operator=operator, device=device)
    loss_model.train()
    compiled_forward = torch.compile(
        lambda carry, values: loss_model(
            carry=carry, batch=values, return_keys=("logits", "q_halt_logits")
        )
    )
    carry = loss_model.initial_carry(batch)
    loss_model.zero_grad(set_to_none=True)
    result = compiled_forward(carry, batch)
    _carry, loss, _metrics, outputs, _finished = result
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("compiled loss is not finite")
    if outputs["logits"].shape != (batch["inputs"].shape[0], batch["inputs"].shape[1], 12):
        raise RuntimeError("compiled lm_head returned an unexpected shape")
    loss.backward()
    parameter_count = assert_finite_gradients(loss_model)
    synchronize(device)
    del loss_model, compiled_forward, result, loss, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return {"operator": operator, "compiled_backward": "PASS", "trainable_parameters_with_gradients": parameter_count}


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    report = Qualification()

    report.run("A_environment", lambda: environment_check(report), required=True)
    dependency_result = report.run("B_dependencies", dependency_check, required=True)
    report.dependencies_ok = dependency_result is not None

    device = torch.device("cuda")
    if report.cuda_ok:
        report.run(
            "C_standalone_mamba2",
            lambda: standalone_mamba2(args.batch_size, device) if report.dependencies_ok else None,
            required=True,
            skip_reason=None if report.dependencies_ok else "Mamba dependencies failed",
        )
    else:
        report.run("C_standalone_mamba2", None, required=True, skip_reason="CUDA environment failed")

    if report.cuda_ok:
        config = make_config(
            batch_size=args.batch_size,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            layers=args.layers,
        )
        batch = make_batch(batch_size=args.batch_size, seq_len=900, device=device)
        attention_result = report.run(
            "D_upstream_attention_trm",
            lambda: trm_cuda_test("attention", config, batch, device, report),
            required=True,
        )
        if report.dependencies_ok:
            hybrid_result = report.run(
                "E_mamba2_attention_trm",
                lambda: trm_cuda_test("mamba2_attention", config, batch, device, report),
                required=True,
            )
        else:
            hybrid_result = report.run(
                "E_mamba2_attention_trm", None, required=True, skip_reason="Mamba dependencies failed"
            )
    else:
        attention_result = report.run(
            "D_upstream_attention_trm", None, required=True, skip_reason="CUDA environment failed"
        )
        hybrid_result = report.run(
            "E_mamba2_attention_trm", None, required=True, skip_reason="CUDA environment failed"
        )
        config = None
        batch = None

    if attention_result is not None and hybrid_result is not None:
        report.run("F_peak_memory", lambda: report.memory, required=True)
    else:
        report.run(
            "F_peak_memory",
            None,
            required=True,
            skip_reason="Both eager TRM runs must pass to compare memory",
        )

    if args.compile and report.cuda_ok and config is not None and batch is not None:
        report.run(
            "G_compile_attention",
            lambda: compile_test("attention", config, batch, device),
            required=False,
        )
        if report.dependencies_ok:
            report.run(
                "G_compile_mamba2_attention",
                lambda: compile_test("mamba2_attention", config, batch, device),
                required=False,
            )
        else:
            report.run(
                "G_compile_mamba2_attention", None, required=False, skip_reason="Mamba dependencies failed"
            )
    else:
        reason = "pass --compile to run the optional compile tests"
        report.run("G_compile_attention", None, required=False, skip_reason=reason)
        report.run("G_compile_mamba2_attention", None, required=False, skip_reason=reason)

    final = {
        "qualification": "FAIL" if report.required_failure else "PASS",
        "environment": report.environment,
        "peak_memory": report.memory,
        "sections": report.sections,
    }
    print("QUALIFICATION_JSON=" + json.dumps(final, sort_keys=True, default=str))
    return 1 if report.required_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
