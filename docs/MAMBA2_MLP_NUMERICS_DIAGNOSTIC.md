# Mamba2-MLP non-finite training diagnostic

The reported exception is raised by `train_one_epoch()` after the model has
completed a full forward pass. The existing loss head computes logits,
language-model loss, and halt loss before that check, so the combined loss alone
cannot identify which operation first became non-finite.

## Current hypothesis

This is a hypothesis, not a confirmed root cause. `Mamba2MLPBlock` in
`src/arc_agi_trm/operators.py` applies pre-norm Mamba-2 and SwiGLU branches,
then adds both updates to the residual stream without normalizing the residual
result. The upstream TRM calls the shared reasoning block repeatedly (`H_cycles`
and `L_cycles`); this run uses 3 H cycles and 4 L cycles with bfloat16
activations. Growth or overflow in one of those recurrent updates could reach
the logits or halt head.

The upstream `ACTLossHead` computes stablemax language loss in float64, but its
halt loss uses `binary_cross_entropy_with_logits` on float32 halt logits. The
trainer checks output-head tensors and individual losses. It checks gradients
after backward, model parameters and optimizer state before updates, and state
immediately around the dense optimizer step. Resume checks checkpoint
structure, configuration, and finite values before loading. The architecture
stability audit adds per-invocation activation and gradient summaries, named
parameter-gradient statistics, and Q-head optimizer-state ranges; see
[`MAMBA2_MLP_ARCHITECTURE_STABILITY_AUDIT.md`](MAMBA2_MLP_ARCHITECTURE_STABILITY_AUDIT.md).

The reported learning rate is consistent with the configured warmup:
`0.0001 * 799 / 2000 = 0.00003995`. That confirms the schedule calculation at
step 799; it does not establish the cause of the failure.

No architecture or optimizer setting has been changed. The first failing
operation must be observed on the Kaggle GPU run before a targeted numerical
fix is selected.

## Four-GPU diagnostic command

From the repository root in Kaggle, with the same dataset used by the failed
run, run this from a fresh diagnostic output directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-nan-diag-seed42 \
  --diagnose-numerics \
  --stop-after-step 900
```

The preset retains seed 42 and all original model, data, optimizer, and
precision settings. The run starts from that seed and stops at the first
completed epoch boundary at or after step 900; a non-finite value stops it
earlier. This reaches beyond the reported step 799 without starting the
100-epoch run. The diagnostic command does not resume or overwrite the original
run directory. It refuses to start if its output directory already contains
`latest.pt` or `best.pt`.

If the failed run itself had resumed from a particular checkpoint and that
continuation must be reproduced, use the same fresh output directory and add
that candidate explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-nan-diag-resume-seed42 \
  --diagnose-numerics \
  --stop-after-step 900 \
  --resume /kaggle/working/outputs/mamba2-mlp-100/latest.pt
```

The trainer checks that the file exists, matches the configured architecture,
seed, data, optimizer, schedule, and world size, and contains finite model,
optimizer, and recurrent-carry tensors before loading any of it. A rejected
candidate is left untouched. A valid candidate is reported as
`resume_checkpoint_validated`; the diagnostic writes to its separate output
directory.

## Reading the output

On the first detected failure, each worker prints a JSON record and appends it
to `numerics_rank0.jsonl` through `numerics_rank3.jsonl` in the diagnostic run
directory. Records include the originating rank, epoch, optimizer step, stage,
tensor name, shape, dtype, non-finite count, and numeric puzzle identifiers
(plus puzzle names when available). Workers synchronize the failure before
gradient collectives, so the originating `FloatingPointError` is visible on
all ranks instead of being replaced by a worker-termination message.

Interpret the earliest record as follows:

- `batch.*`: invalid token, target, or puzzle identifier in the data batch.
- `forward.*block_input`, `mamba_norm_input`, `mamba_output`, `mamba_residual`,
  `mlp_norm_input`, `mlp_output`, or `mlp_residual`: first bad Mamba/MLP block
  boundary or stage. `block_input` checks the result of the upstream recurrent
  input injection; compare the first bad branch output with its residual result.
- `forward.*lm_head.output` or `forward.*q_head.output`: recurrent blocks were
  still finite at the last observed boundary; the corresponding output head
  first produced a bad value.
- `loss_component.*` or `loss.total`: logits reached the loss head, and the
  named component or combined loss is the first observed failure.
- `loss.reduced_global_mean`: local losses were finite but their distributed
  reduction or per-example mean was not.
- `gradient.*`: backward produced a bad gradient. The stage distinguishes the
  local backward result from the gradient after cross-rank reduction.
- `model.parameter.*`, `model.buffer.*`, or `optimizer[*].state.*` at
  `before_optimizer_update`: corruption existed before the next optimizer step.
- The same tensor names at `after_optimizer_update`: the update created the
  first bad parameter or optimizer-state value.
- `resume_checkpoint_rejected` or `checkpoint_numerical_failure`: a candidate
  checkpoint was rejected or a save was prevented. These records include the
  rank, checkpoint epoch/step, state tensor metadata, and carried puzzle IDs.

If the run reaches the boundary at or after step 900 without a diagnostic
record, the reported failure was not reproduced from the configured seed and
starting state in that run. Preserve the environment report, config snapshot,
dataset manifest, and output logs before changing settings.

## Follow-up runs

After the first failing operation has been identified and any necessary
targeted code fix has been made, run a bounded stability pass in a separate
directory. This repeats the same seeded configuration through at least 900
optimizer steps without the extra per-tensor instrumentation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-stability-seed42 \
  --stop-after-step 900
```

Only after that bounded run completes without a numerical failure should the
100-epoch experiment be started, again in a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-100-confirmed
```
