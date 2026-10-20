# Reproducible Mamba-2 hybrid TRM experiments

This runbook prepares two experiments on the public ARC-AGI-2 training set.
The full training runs are intentionally not started by the implementation or
test commands.

## Repository and source audit

At implementation start, `git status --short --branch` was clean at
`5020cf6` (`master`, tracking `origin/master`). The upstream TRM source under
`external/TinyRecursiveModels/` is present as individually tracked files. The
Git index also contains a `160000` gitlink at `trm-temp`, but there is no
`.gitmodules` mapping; `git submodule status` therefore fails with “no submodule
mapping found for path 'trm-temp'”. This pre-existing Git metadata was left
untouched. The Kaggle setup should use the checked-in `external/` tree or an
attached copy selected with `TRM_UPSTREAM_ROOT`.

The factory replaces only TRM's per-step block list. The vendored recursive
model, ACT wrapper, carries, token embedding, loss, and heads remain unchanged.
`operator=mamba2_attention` installs two Mamba-2 mixers, the existing
non-causal attention mixer, and the existing SwiGLU MLP in each shared block.
The factory raises on missing Mamba dependencies or a failed block install;
it has no attention fallback.

## Architecture and optimization

Both presets load `configs/mamba2_hybrid_base.json`; only experiment name,
output directory, and epoch budget differ.

| Setting | Value |
|---|---:|
| Hidden dimension | 512 |
| Grid sequence | 900 tokens; 916 positions including 16 puzzle-prefix positions |
| Vocabulary | 12: PAD 0, EOS 1, colors 2–11 |
| Outer / inner recursion | `H_cycles=3`, `L_cycles=4` |
| Shared TRM layers | `L_layers=2`; `H_layers` is ignored upstream |
| Attention | 8 heads, head dimension 64, non-causal |
| Mamba-2 per block | 2 operators; `d_state=128`, `expand=2`, `headdim=64`, `d_conv=4`, `d_ssm=1024` (`None` means full inner width), `ngroups=1` |
| Mamba-2 kernel/norm settings | gated RMSNorm enabled, norm-before-gate off, chunk 256, memory-efficient path enabled, projection bias off, convolution bias on |
| MLP | SwiGLU, expansion 4 |
| ACT | max 16 supervision steps; halt exploration 0.1; no continue-Q loss |
| Per-device / global batch | 8 per L4 × 4 GPUs = 32 global |
| Gradient accumulation | 1; effective global batch 32 |
| Dense optimizer | upstream `AdamATan2`; LR `1e-4`, betas `(0.9, 0.95)`, weight decay `0.1` |
| Puzzle-embedding optimizer | upstream distributed sparse SignSGD; LR `1e-2`, weight decay `0.1` |
| LR schedule | 2,000 update warmup; minimum ratio 1.0, so LR stays at base after warmup |
| Gradient clipping | disabled, matching the upstream loop |

The shared two-block `L_level` is called 15 times in each inner supervision
step: 10 calls run under `no_grad` and 5 run with gradients. Each call applies
two hybrid blocks, so a step has 30 block applications, 20 without gradients
and 10 with gradients. TRM carries `z_H`, `z_L`, ACT steps, halted flags, and
the current batch forward; checkpoints preserve these carries.

The other configured Mamba-2 details follow the pinned `mamba-ssm==2.3.2.post1`
API: `d_ssm=None` uses the full 1,024-wide inner state, which gives 16 SSM
heads; `ngroups=1`; `rmsnorm=True`; `norm_before_gate=False`; `chunk_size=256`;
`use_mem_eff_path=True`; `bias=False`; and `conv_bias=True`. The pinned API's
remaining defaults include `A_init_range=(1,16)`, per-head `D`, SiLU
activation, and `dt` initialization from 0.001 to 0.1 with a 1e-4 floor. These
values are defined in the [tagged Mamba-2 constructor](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/mamba_ssm/modules/mamba2.py).

## Data, split, augmentation, and sampling

The builder reads only `arc-agi_training_challenges.json` and
`arc-agi_training_solutions.json`. It verifies that there are 1,000 tasks and
that every public test input has a solution. A seed-42 task-level split holds
out 100 tasks for validation and uses the other 900 for training. Task IDs and
source-file SHA-256 digests are saved in `dataset_manifest.json`; no public
evaluation or private test data is read.

For each training task, the upstream transformation builder keeps the original
and attempts up to 1,000 unique variants. It deduplicates symmetric variants,
so the actual count can be lower. Variants use the existing eight rotations
and reflections, color permutations of 1–9 with black fixed, and training
translations with one pair left untranslated. The validation side contains
only each held-out task's original public test input/output pairs and has no
augmented variants or translations.

For the bundled 1,000-task public training file with seed 42, the split is 900
training tasks and 100 validation tasks. Those 900 training tasks contain
3,877 unaugmented supervised input/output pairs (demonstrations plus labeled
public training test pairs); the 100 validation tasks contain 106 public test
pairs. If every requested unique augmentation is generated, training stores
up to 900,900 task variants and 3,880,877 training pairs, plus the 100 original
validation variants and 106 validation pairs. Symmetric-task deduplication can
lower the training variant and pair totals; the generated dataset manifest is
the authoritative count for the attached Kaggle files.

The builder writes input and label `.npy` arrays one task group at a time.
It does not keep the full augmented dataset's grids in memory. The manifest
records original tasks, actual variants, generated augmented variants,
unaugmented input/output pair counts, and stored augmented pair counts.

The unchanged `PuzzleDataset` sampler shuffles each original training-task
group once per epoch, chooses one stored variant from that group, then packs
that variant's grid pairs into global batches. It does **not** process every
augmented variant in every epoch. It drops the final incomplete global batch;
the upstream packer may also use only the portion of a task variant that fits
at a batch boundary. Therefore, the exact number of examples emitted in an
epoch is logged from the actual batches as `training_examples_sampled`.
Distributed ranks use a shared NumPy sampler stream so they slice the same
global batch. If a smaller per-device batch is required, sparse embedding
gradients are accumulated together with their matching task IDs before the
upstream SignSGD update.
Optimizer updates are logged separately as `optimizer_step`; neither value is
the number of stored variants or the number of original ARC tasks.

## Build the data on Kaggle

Attach the public ARC Prize 2026 competition dataset and the already-qualified
Mamba/causal-conv wheels. Follow the offline install order in
[`KAGGLE_MAMBA_INSTALL.md`](KAGGLE_MAMBA_INSTALL.md), then run the environment
and kernel checks before training:

```bash
python scripts/audit_environment.py
python scripts/kaggle_gpu_qualification.py --batch-size 1 --h-cycles 3 --l-cycles 4 --layers 2
```

The first command should see Python 3.12.13, the supplied PyTorch/CUDA build,
four L4 GPUs, and the expected ABI. The second checks standalone Mamba kernels
and a full-width/full-recursion hybrid forward and backward at batch 1. It is
a kernel qualification, not the batch-8 memory benchmark.

Build 1,000 training tasks with the task-level 100-task validation holdout:

```bash
python scripts/build_arc_agi2_from_kaggle.py \
  --source-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-2 \
  --output-dir /kaggle/working/data/arc2-mamba2-aug1000 \
  --num-aug 1000 --validation-tasks 100 --seed 42
```

The build rejects any task count other than 1,000 by default and refuses to
overwrite a non-empty output directory. It prints actual augmentation and pair
counts at completion. Building the full million-variant upper bound is a
substantial CPU and disk operation; it is a separate preparation command, not
part of either training run.

## Full-model benchmark and runtime estimate

Run this after the batch-1 qualification and dataset build. It uses the actual
training sampler, per-device batch 8, full 512-dimensional model, 900-token
grids, configured recursion, sparse and dense optimizers, the full validation
split, and a full-sized checkpoint write. It reports warmup time, steady-state
training examples/second, validation duration, checkpoint-writing time, and
peak allocated/reserved memory. The resulting `benchmark.json` derives
100-epoch and 1,000-epoch estimates from those measurements and states its
sampler/throughput assumptions.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/benchmark_mamba2_hybrid.py \
  --config configs/mamba2_hybrid_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --output-dir /kaggle/working/benchmarks/mamba2-hybrid-full \
  --warmup 2 --steps 5
```

If batch 8 does not fit, the command fails with the CUDA error; change the
shared `global_batch_size` to a multiple of four (for example 16, which is
per-device 4) and set `gradient_accumulation_steps` to 2 to keep the effective
global batch at 32. Repeat the benchmark before launching training. Do not use
the earlier reduced-model throughput as a substitute. This workspace has
CPU-only PyTorch, so the full
model throughput, L4 memory, validation time, checkpoint time, and runtime
estimates are currently unverified.

## Train and resume

The exact four-L4 run commands are:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_hybrid_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-hybrid-100
```

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_hybrid_1000_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-hybrid-1000
```

Runs write to separate directories under `/kaggle/working/outputs/`.
`latest.pt` is saved at epoch 0, every 10 completed epochs, and at the final
epoch. `best.pt` points to the best validation checkpoint. A checkpoint
contains strict model weights, both optimizer states, scheduler step and
horizon, completed epoch, per-rank Python/NumPy/PyTorch/CUDA RNG states, sampler
iteration, TRM carry state, and cumulative epoch timing. Its fingerprint binds
the run settings to the exact dataset manifest and source file hashes. Resume
requires the same GPU count and matching run configuration:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_hybrid_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-hybrid-100 --resume latest
```

Replace the 100-epoch config with the 1,000-epoch config to resume Experiment
B. The script does not continue after a failed update or mark an incomplete
epoch complete. An interrupt resumes from the previous valid checkpoint.

## Validation and reporting

Validation runs every 10 epochs and at the final epoch. It uses one greedy
top-1 output per official training-task test input, with the model's full 16
inference steps. A validation task counts as solved only when all its test
outputs exactly match. `validation_grid_exact_accuracy` and
`validation_task_exact_match` are separate metrics. The attempt budget is one;
there is no pass@2 selection in this run. The report includes task count,
solved count, grid count, validation loss, inference duration, checkpoint path,
seed, per-device/global/effective batch sizes, examples/second, average
training-plus-validation epoch duration, checkpoint-writing time, and GPU
memory. `metrics.jsonl` is machine-readable; `run.log` is human-readable.

Evaluate the best checkpoint separately on one L4 (batch size 8):

```bash
python scripts/evaluate_mamba2_hybrid.py \
  --config configs/mamba2_hybrid_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --checkpoint /kaggle/working/outputs/mamba2-hybrid-100/best.pt \
  --batch-size 8
```

## Local verification

CPU unit tests use an injected trainable fake mixer only for architecture and
training-loop tests. They do not validate the Mamba-2 CUDA extension. In this
workspace:

```powershell
.venv\Scripts\python.exe -m compileall -q src scripts tests
.venv\Scripts\python.exe -m pytest -q
```

The benchmark and full training commands require CUDA, `mamba-ssm`,
`causal-conv1d`, the upstream sparse SignSGD code, and a working
`adam_atan2_backend`. No successful full training run is claimed here.
