# Original TRM baseline and training progress

## Architecture presets

`configs/trm_baseline_100_epochs.json` selects `operator: "attention"` and uses
the vendored upstream TRM block implementation. Its model settings match
`external/TinyRecursiveModels/config/arch/trm.yaml`: H/L cycles 3/6, two shared
blocks, hidden size 512, eight heads, expansion 4, RoPE, bf16, 16 puzzle
embedding tokens, and the upstream ACT defaults. Its optimizer settings follow
the upstream pretraining config. The 100-epoch budget, ARC-AGI-2 dataset split,
and checkpoint/evaluation intervals are the project run policy.

`configs/trm_matched_mamba2_mlp_100_epochs.json` layers over the existing
Mamba-2 MLP base and changes the operator to `attention`. It keeps the Mamba
experiment's seed, dataset, batch size, model dimensions, schedule, and other
applicable settings. Its Mamba-specific config values remain unused; selecting
`attention` returns the upstream model before importing the project Mamba
operator module.

The CLI already selects architecture through the config's `operator` field:

- `attention`: original upstream TRM, with its original `L_level.layers`.
- `mamba2_attention` or `mamba2_mlp`: existing Mamba replacements.

No Mamba package is imported or block instantiated by `operator: "attention"`.
The trainer still uses the upstream sparse embedding optimizer and the
configured dense optimizer, which are independent of Mamba.

## Four-GPU Kaggle launch

From the repository root, run the faithful upstream baseline with fresh state:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_mamba2_hybrid.py \
  --config configs/trm_baseline_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/trm-baseline-100
```

This command does not pass `--resume` or `--init-from`. Do not start it until
the selected output directory is available for a fresh run. The matched
comparison uses `configs/trm_matched_mamba2_mlp_100_epochs.json` and a separate
`--run-dir`.

## Progress and ETA

Rank zero owns one tqdm bar; other ranks do not create a progress display. The
bar advances only after a completed optimizer update, including a short final
gradient-accumulation window. It shows epoch, optimizer step/total, the current
accumulation-window loss, learning rate, recent examples per second, elapsed
time, and a rolling `eta~` estimate. Grid and exact accuracy are labeled
`grid(last)` and `exact(last)` and refer to the last completed epoch.

The upstream `PuzzleDataset` is iterable-only and has no loader length. The
trainer therefore counts the exact yields from the upstream sampler's group
indices, per-epoch Philox shuffle, puzzle selection, batch packing, and final
partial-batch drop. It then converts batches to updates with
`ceil(batch_count / gradient_accumulation_steps)` for each epoch. This planning
pass does not consume the sampler's process-global NumPy RNG or advance its
resume counter. The total honors the epoch budget and the existing
`--stop-after-step` epoch-boundary behavior. Observed epoch update counts revise
the ETA plan if a future sampler change makes the count differ. ETA is
explicitly approximate and uses the rolling duration of recent optimizer
updates; compilation, optimizer warmup, checkpointing, validation, and slower
early updates affect the estimate as their time enters the rolling window. The
bar starts after model setup and the initial checkpoint, and shows `eta~=?`
until it has observed an optimizer update.

Epoch JSON metrics remain in `metrics.jsonl` and `run.log`. The normal progress
bar does not show tensor or per-layer diagnostics.

## CPU smoke test

Run this short test from the repository root:

```bash
.venv/Scripts/python.exe -m pytest -q tests/test_trm_baseline.py
```

It blocks Mamba imports while constructing the upstream attention model, runs
one CPU forward/backward/optimizer update with finite loss and gradients,
exercises the progress display interface, round-trips both training/resume and
model-only inference checkpoints, and compares planned optimizer updates with
actual upstream sampler epochs. The progress unit test uses a tiny
tqdm-compatible test double; actual training uses the pinned `tqdm==4.67.1` dependency in
`requirements/kaggle-training.txt`.

## Existing runs and numerical diagnostics

The source edits affect newly launched Python processes. A running Python
process already has its imported model and trainer code in memory; this work
does not signal, edit, or stop it. On Kaggle, inspect a live run without
interrupting it:

```bash
ps -eo pid,etime,args | grep '[t]rain_mamba2_hybrid.py'
PID=1234  # replace with the PID from the previous command
ps -p "$PID" -o pid,etime,args
tr '\0' ' ' < "/proc/$PID/cmdline"; echo
```

Use the command's `--run-dir` and `--config` values to check its recorded
identity (set `RUN_DIR` to the process's `--run-dir`, or its config's default):
For older processes launched with a `--run-dir` override, the command line is
authoritative because their snapshot may contain only the preset's default
directory.

```bash
RUN_DIR=/kaggle/working/outputs/trm-baseline-100
jq '{name: .experiment.name, operator: .experiment.operator, run_dir: .experiment.run_dir}' \
  "$RUN_DIR/config_snapshot.json"
tail -n 1 "$RUN_DIR/metrics.jsonl"
ls "$RUN_DIR"/numerics_rank*.jsonl 2>/dev/null
```

Diagnostic JSONL files are created only when `--diagnose-numerics` is enabled.
That opt-in mode checks tensors, gradients, parameters, optimizer state, and
distributed synchronization, and can substantially reduce training
throughput. Leave the flag off for normal performance runs; it is off by
default.
