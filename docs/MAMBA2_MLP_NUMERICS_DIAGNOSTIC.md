# Mamba2-MLP non-finite training diagnostic

## Shared Q-head incident update

The reported attention TRM failure at optimizer step 905 has the same Q-head
optimizer-state signature as the earlier Mamba2-MLP failures. That evidence
shifts the first investigation to the shared ACT loss and dense optimizer
path; recurrent architecture growth remains possible but is not established.

The vendored `ACTLossHead` and `TinyRecursiveReasoningModel_ACTV1` use the
upstream semantics: `q_head(z_H[:, 0])` produces halt and continue **logits**;
the halt target is whole-sequence correctness; BCE-with-logits uses `sum`
reduction without a halt mask; and the total is
`lm_loss + 0.5 * (q_halt_loss + q_continue_loss)`. `target_q_continue` is
computed inside `torch.no_grad()`. With the repository presets'
`no_ACT_continue: true`, the continuation target and continuation loss are
absent, and halt decisions use `q_halt_logits > 0`. Halted examples reset their
carry on their next forward. Source inspection also found an upstream tuple
arity bug in the continuation-target branch: `inner.forward()` returns three
values, but the wrapper unpacks five. That branch is disabled in the current
presets, but failed whenever continuation ACT was enabled. The vendored call
now consumes its actual three-value return and has a CPU regression test. No
loss weighting or target definition was changed.

In this trainer, the tqdm `Loss avg30` is a batch-size-weighted rolling mean of
the last 30 optimizer updates. It is derived from each raw loss passed to
backward, so it is not the raw batch scalar and is not an aggregation bug. The
new `training_diagnostics_rank0.jsonl` records that raw scalar and every loss
component for the configured detail window. It also records output-logit
finite counts/maxima, Q logits and targets, valid-token masks and denominators,
Q-head weights/activations/activation gradients, and pre/post-update parameter
and optimizer-state summaries. `propagation_rank0.jsonl` has the detailed
per-parameter gradients and optimizer settings for those steps.

The pinned AdamATan2 CUDA kernel contains
`exp_avg_sq = lerp(exp_avg_sq, grad * grad, mbeta2)`. For float32 Q-head
parameters, a finite gradient above about `1.84e19` can overflow at `grad *
grad`. The diagnostic trainer now checks this bound before calling the dense
optimizer and records the offending parameter and gradient magnitude. This is
a concrete candidate mechanism for finite gradients corrupting `exp_avg_sq`,
but the reported run's gradient values are not available in this checkout, so
it is not yet confirmed as the step-905 root cause. The original failed-run
logs/checkpoints were also not present under this workspace's `outputs/`.

The local environment is CPU-only and its installed Python wrapper cannot
import the compiled `adam_atan2_backend`; CPU tests therefore validate the
pre-step guard and diagnostics, not the CUDA kernel. The Kaggle runs below
must establish whether the original step-905 failure reaches the square
overflow guard, starts with invalid gradients/state, or first becomes invalid
inside the optimizer update.

### Matched attention and Mamba2-MLP capture

Run both commands from fresh directories on the same four-GPU Kaggle session
and dataset. Both presets use seed 42, global batch 32, the same warmup and
optimizer settings, and the same training data order. Neither command resumes
from an unvalidated checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/trm_matched_mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/trm-qhead-step905-attention \
  --diagnose-numerics --diagnose-step-range 900:905 --stop-after-step 905

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/trm-qhead-step905-mamba2-mlp \
  --diagnose-numerics --diagnose-step-range 900:905 --stop-after-step 905
```

`--stop-after-step` stops at the first completed epoch boundary at or after
step 905. The model writes a step-zero checkpoint in each new directory and
keeps the existing failed-run artifacts untouched. Do not use `--resume` until
both model parameters and optimizer states in that checkpoint have passed
validation.

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
