# Original TRM baseline and training progress

## Why the first run used batch 192 per GPU

`global_batch_size` is the aggregate microbatch across all distributed ranks.
The trainer resolves it as:

```text
per_device_batch_size = global_batch_size / world_size
effective_global_batch_size = global_batch_size * gradient_accumulation_steps
```

The failed Kaggle run used `global_batch_size: 768` with four ranks, so the
standalone attention TRM received 192 examples per GPU. With accumulation 1,
its effective update batch was 768. The current faithful architecture preset
sets global batch 64 and expects 32 examples per GPU. At four ranks it
resolves to 16 per GPU, so startup rejects this mismatch until those values
are aligned. `model.batch_size` is derived from the resolved per-device value
because the upstream model uses it to size its local sparse-embedding
buffers. Startup prints the resolved values before model construction.

The failed run's 192-example allocation explains the reported CUDA OOM during
the RoPE attention forward. The architecture-faithful preset remains
separately identifiable as `trm-upstream-faithful-100`; the matched preset
uses a smaller microbatch for the L4 comparison.

## Matched standalone TRM preset

Use `configs/trm_matched_mamba2_mlp_100_epochs.json` for the stable comparison
run. It layers over the faithful upstream TRM preset and changes the seed to
42, L cycles to 4 to match the Mamba2-MLP experiment, and the batch settings.
The dataset, model dimensions, optimizer values, epochs, and evaluation policy
match the Mamba2-MLP experiment where applicable. It keeps the original TRM
RoPE setting and upstream attention blocks, normalization, initialization,
recurrence, carry, ACT/halting, sparse puzzle embedding, and output heads.
Selecting `operator: "attention"` builds the vendored upstream TRM model and
does not import or instantiate a Mamba block.

Resolved settings with four GPUs:

| Setting | Matched preset |
| --- | ---: |
| World size | 4 |
| Per-device batch size | 8 |
| Global microbatch | 32 |
| Gradient accumulation | 1 |
| Effective global batch per optimizer update | 32 |
| Seed | 42 |
| Epochs | 100 |
| Position encoding | Upstream RoPE |

If the four-GPU smoke test still OOMs at 8 per device, retry with
`configs/trm_matched_mamba2_mlp_100_epochs_bs4.json`. That explicit fallback
uses per-device batch 4, global microbatch 16, and accumulation 2, keeping the
effective update batch at 32. It does not change batch sizes after a run
starts.

## Four-GPU smoke test

Run from the repository root on the four-GPU Kaggle machine. The smoke option
does exactly two optimizer updates, skips validation and checkpoint writes,
and records the maximum allocated and reserved memory across ranks after each
update in `smoke_memory.jsonl`. `metrics.jsonl` and `run.log` receive a
structured completion record. Use a fresh smoke output directory.

```bash
torchrun --standalone --max-restarts=0 --nproc_per_node=4 scripts/train_mamba2_hybrid.py \
  --config configs/trm_matched_mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/trm-baseline-smoke-bs8 \
  --smoke-test-steps 2
```

If that reports CUDA OOM, retry the smaller configured microbatch:

```bash
torchrun --standalone --max-restarts=0 --nproc_per_node=4 scripts/train_mamba2_hybrid.py \
  --config configs/trm_matched_mamba2_mlp_100_epochs_bs4.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/trm-baseline-smoke-bs4 \
  --smoke-test-steps 2
```

Both commands use `torchrun` with restarts disabled. A rank that hits CUDA OOM
reports its rank and resolved batch, exits with failure, and lets the torchrun
worker agent stop and reap the other local ranks. The trainer also destroys
the process group during normal exception cleanup. No allocator setting is
used as a substitute for reducing the configured microbatch.

## Full comparison run

After the smoke test completes without OOM, the full matched run command is:

```bash
torchrun --standalone --max-restarts=0 --nproc_per_node=4 scripts/train_mamba2_hybrid.py \
  --config configs/trm_matched_mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/trm-baseline-matched-100
```

The 100-epoch command is documented but has not been launched. The faithful
architecture preset remains available at
`configs/trm_baseline_100_epochs.json`. Its current global batch 64 and
expected per-device batch 32 are inconsistent at world size 4, so it is
rejected at startup; align those values before using that preset. The earlier
Kaggle run used global 768 and resolved to 192 per device.

## Progress total and logs

The upstream `PuzzleDataset` is iterable-only and has no loader length. Before
constructing the model, the trainer counts the exact upstream sampler yields
from the dataset group indices, per-epoch Philox shuffle, puzzle selection,
batch packing, and dropped final partial batch. It converts each epoch's batch
count to optimizer updates with
`ceil(batch_count / gradient_accumulation_steps)`. The global batch passed to
the loader already reflects the resolved world-size plan. It refuses to
create a progress bar if any epoch has zero planned updates, so a broken plan
cannot display `0/0`. The plan is included in the startup batch-plan line and
`config_snapshot.json`; each completed epoch checks the prediction against the
actual loader and corrects later totals if needed.

Only rank zero creates tqdm. It displays epoch/total epochs, completed/total
optimizer steps, rolling training loss, training grid and exact accuracy,
learning rate, recent examples per second, elapsed time, and approximate ETA.
After validation completes, it separately displays validation loss and
validation task exact match. Gradient accumulation's final short window counts
as an optimizer update. Full epoch summaries remain in `metrics.jsonl` and
`run.log`; notebook output contains no epoch-metrics JSON dump.

## Local verification

Run the CPU tests from the repository root:

```bash
.venv/Scripts/python.exe -m pytest -q tests/test_trm_baseline.py
```

These tests exercise the original upstream attention block without importing
Mamba, run a CPU forward/backward/optimizer update, check progress fields,
round-trip checkpoints, compare planned counts with actual upstream sampler
epochs, validate matched and fallback batch plans, and reject inconsistent
batch expectations. The real
four-GPU forward/backward memory smoke test and CUDA OOM worker shutdown still
need to run on Kaggle. Do not launch the 100-epoch command until the smoke test
passes.
