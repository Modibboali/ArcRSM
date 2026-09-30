"""Benchmark attention and Mamba-2 hybrid TRMs on a Kaggle CUDA GPU."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import statistics
import sys
import time
import traceback

import torch

from kaggle_trm_runtime import (
    build_loss_model,
    make_batch,
    make_config,
    memory_report,
    model_step,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=900)
    parser.add_argument("--h-cycles", type=int, default=2)
    parser.add_argument("--l-cycles", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--forward-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def benchmark_operator(
    operator: str,
    *,
    config: dict,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    model = build_loss_model(config, operator=operator, device=device)
    model.train()

    def forward_once() -> None:
        with torch.no_grad():
            model_step(model, batch, return_outputs=False)

    def train_once() -> None:
        model.zero_grad(set_to_none=True)
        _carry, loss, _metrics, _outputs, _finished = model_step(model, batch)
        loss.backward()

    try:
        for _ in range(warmup):
            forward_once()
            train_once()
        synchronize(device)

        def measure(fn) -> list[float]:
            samples = []
            for _ in range(repeats):
                synchronize(device)
                start = time.perf_counter()
                fn()
                synchronize(device)
                samples.append((time.perf_counter() - start) * 1000.0)
            return samples

        forward_ms = measure(forward_once)
        forward_backward_ms = measure(train_once)

        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        train_once()
        synchronize(device)
        memory = memory_report(device)
        return {
            "operator": operator,
            "forward_latency_ms_median": statistics.median(forward_ms),
            "forward_latency_ms_samples": forward_ms,
            "forward_backward_latency_ms_median": statistics.median(forward_backward_ms),
            "forward_backward_latency_ms_samples": forward_backward_ms,
            "forward_backward_peak_memory": memory,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.seq_len <= 0:
        raise SystemExit("--batch-size and --seq-len must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("--warmup must be nonnegative and --repeats must be positive")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is unavailable; run this benchmark in the Kaggle GPU notebook.")
        return 1

    try:
        from mamba_ssm.modules.mamba2 import Mamba2  # noqa: F401

        versions = {
            "mamba-ssm": importlib.metadata.version("mamba-ssm"),
            "causal-conv1d": importlib.metadata.version("causal-conv1d"),
        }
        device = torch.device("cuda")
        config = make_config(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            layers=args.layers,
            forward_dtype=args.forward_dtype,
        )
        batch = make_batch(batch_size=args.batch_size, seq_len=args.seq_len, device=device)
        results = {}
        for operator in ("attention", "mamba2_attention"):
            print(f"[RUN] {operator}: batch={args.batch_size}, sequence={args.seq_len}")
            results[operator] = benchmark_operator(
                operator,
                config=config,
                batch=batch,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            print("[PASS] " + json.dumps(results[operator], sort_keys=True))
        report = {
            "status": "PASS",
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "mamba_versions": versions,
            },
            "configuration": {
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "h_cycles": args.h_cycles,
                "l_cycles": args.l_cycles,
                "layers": args.layers,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "forward_dtype": args.forward_dtype,
            },
            "results": results,
        }
        print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))
        return 0
    except BaseException as exc:
        print(f"[FAIL] benchmark: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        print("BENCHMARK_JSON=" + json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
