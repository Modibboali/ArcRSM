# Mamba2-MLP architecture stability audit

## Source trace

`build_trm()` in [`src/arc_agi_trm/upstream.py`](../src/arc_agi_trm/upstream.py)
first creates the upstream TRM and then replaces only
`model.inner.L_level.layers`. The upstream recursion and output heads remain in
[`external/TinyRecursiveModels/models/recursive_reasoning/trm.py`](../external/TinyRecursiveModels/models/recursive_reasoning/trm.py).
`H_layers` is ignored by the upstream config. `L_layers` controls the one shared
`ModuleList` used for every latent update.

For the reported `L_layers=2`, two distinct `Mamba2MLPBlock` objects are
created. Each contains one distinct Mamba-2 module and one distinct SwiGLU
module, for two Mamba-2 mixers and two MLPs total. There is no module cloning
or per-cycle construction. Both blocks are reused across all H and L updates.
The LM head and Q head are the original single upstream modules.

The forward schedule for `H_cycles=3`, `L_cycles=4` is:

```text
carry z_H, z_L (detached from the previous model call)
input_embeddings

H cycle 0, torch.no_grad():
  L0: z_L <- L_level(z_L, z_H + input_embeddings)
  L1: z_L <- L_level(z_L, z_H + input_embeddings)
  L2: z_L <- L_level(z_L, z_H + input_embeddings)
  L3: z_L <- L_level(z_L, z_H + input_embeddings)
  H:  z_H <- L_level(z_H, z_L)

H cycle 1, torch.no_grad(): same four L updates, then one H update

H cycle 2, gradients enabled:
  L0, L1, L2, L3: same four L updates
  H: z_H <- L_level(z_H, z_L)

new carry = detach(z_H), detach(z_L)
logits = lm_head(z_H), with the 16 puzzle-prefix positions removed
q logits = q_head(z_H[:, 0])
```

Every `L_level` call first adds its `input_injection`, then executes each
shared block in order. For the L updates the injection is `z_H + input_embeddings`;
for the H update it is `z_L`. The first two H cycles execute under
`torch.no_grad()`. Only the final H cycle has an autograd graph. The returned
carry is detached before the next outer training call, so gradients do not
cross ACT calls or batches. LM and Q losses do backpropagate through the final
H cycle to its input embeddings and shared blocks.

### Differentiable depth

There are `L_cycles + 1 = 5` gradient-bearing `L_level` calls in the final H
cycle: four L updates and one H update. Each call passes through all
`L_layers` shared blocks. Thus, for one training model call:

| Configuration | Differentiable block invocations | Mamba branches | SwiGLU branches |
| --- | ---: | ---: | ---: |
| A: `L_layers=2`, `L_cycles=4` | 5 × 2 = **10** | 10 | 10 |
| B: `L_layers=1`, `L_cycles=4` | 5 × 1 = **5** | 5 | 5 |

Each block has two residual sublayers. Counting just block calls would
understate the number of differentiable branch transforms: A has 20 (10
Mamba-2 and 10 SwiGLU), while B has 10. Earlier no-gradient uses still execute
the same parameters and can affect the final hidden state, but contribute no
backward path. The relevant loop is `TinyRecursiveReasoningModel_ACTV1_Inner.forward`,
not the product `H_cycles * L_cycles * L_layers`.

## Stability-relevant implementation details

`Mamba2MLPBlock.forward()` in [`operators.py`](../src/arc_agi_trm/operators.py)
uses pre-normalization and unscaled residual additions:

```text
h1 = h + Mamba2(RMSNorm(h))
h2 = h1 + SwiGLU(RMSNorm(h1))
```

There is no post-normalization, residual multiplier, or residual scaling after
either branch. RMSNorm computes its variance in float32 and casts its result
back to the input dtype. The residual stream and Mamba path use the configured
bfloat16 forward dtype. The upstream attention block instead applies
post-normalization after attention-plus-residual and again after
SwiGLU-plus-residual. For the same model config, both architectures retain the
same configured `L_level` layer count, cycle schedule, input injection, carry
detach, LM head, and Q-head connectivity.

Compared with the vendored upstream `trm.yaml` defaults, this preset keeps
`H_cycles=3`, `L_layers=2`, `H_layers=0` (ignored), and `hidden_size=512`, but
uses `L_cycles=4` instead of 6 and `pos_encodings="none"` instead of RoPE. The
Mamba scan therefore runs over the flattened token stream without the baseline
block's explicit rotary position transform. The depth calculation above uses
the actual Mamba preset values. At the upstream default of six L cycles, the
two-layer model would have seven gradient-bearing `L_level` calls, or 14 block
invocations; that is not the reported run's schedule.

With the Mamba preset, each mixer uses `d_model=512`, `expand=2`, `headdim=64`,
`d_state=128`, `ngroups=1`, `d_conv=4`, `d_ssm=null`, gated RMSNorm, and the
fused memory-efficient scan. This implies `d_inner=d_ssm=1024` and 16 SSM heads.
The block has no attention and no additional MLP inside Mamba because
`d_ssm=d_inner`.

The pinned Mamba-2 source uses `A=-exp(A_log)`, applies softplus to the time
step, and uses the discretized decay `exp(dt*A)` in its scan/update. Its Mamba
gate and gated RMSNorm are internal to the mixer. The module initializes
`A_log`, `dt_bias`, and `D` specially; its projection and convolution weights
use PyTorch layer initialization. This differs from the upstream attention
and SwiGLU projections, which use the TRM's truncated LeCun initializer.
The Mamba block's `rmsnorm=True` does not normalize the outer residual stream.
See the version-pinned
[Mamba-2 implementation](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/mamba_ssm/modules/mamba2.py).

The LM head retains its upstream initialization. The Q head weight is
explicitly zeroed and its bias set to `-5`; it reads `z_H[:, 0]`, the first
puzzle-prefix position. The failed state shape `[2, 512]` matches that Q-head
weight. In the current training code `optimizer[0]` is distributed sparse
SignSGD for puzzle embeddings and `optimizer[1]` is `AdamATan2` for all dense
model parameters, including the Q head. Gradient clipping is disabled by the
current config (`gradient_clip_norm: null`). The local Python environment has
the AdamATan2 wrapper but not its CUDA backend or `mamba-ssm`.

These details make residual growth a plausible hypothesis, but they do not
establish the cause of the reported `exp_avg_sq` corruption. A non-finite or
extremely large Q-head gradient, pre-existing optimizer state, or the optimizer
kernel itself must be distinguished from recurrent activation growth.

## Instrumentation added

With `--diagnose-numerics`, the trainer now writes a per-rank
`propagation_rankN.jsonl` beside the existing `numerics_rankN.jsonl`. The
propagation log stores summaries only, not tensor contents.

- Each block invocation is labeled with its H cycle, L update or H update,
  `L_level` call index, layer index, and whether autograd is enabled.
- Each invocation records hidden input/output RMS, standard deviation, maximum
  absolute value, finite count, plus the Mamba and MLP normalized inputs,
  branch outputs, and residual outputs.
- Tensor gradient RMS, L2 norm, maximum absolute value, and finite count are
  attached to each differentiable invocation. Activation and gradient trends
  are summarized by cycle and shared layer.
- Immediately before the dense update, the log records total model gradient
  norm, every named parameter's gradient summary, nested module gradient
  groups, optimizer parameter-group norms, and the Q-head weight gradient.
  Because block parameters are shared across invocations, parameter gradients
  are the aggregate from the full backward pass; per-invocation activation and
  branch-output gradients show how the signal changes through the recursion.
- The Q-head `exp_avg` and `exp_avg_sq` receive finite count and min/max/RMS/L2
  summaries immediately before and after `optimizer[1].step()`. The existing
  diagnostics also check model parameters and all optimizer state before and
  after updates. This distinguishes an already-invalid state from one made
  invalid during the update.

Use the earliest recorded stage to classify the failure:

| Observation | Interpretation to test |
| --- | --- |
| Non-finite gradient before update | Case 1: backward already produced invalid gradients. |
| All gradients finite, with very large `max_abs` or norms | Case 2: finite but extreme gradients; compare Q-head and model-wide values. |
| Pre-step state and gradient finite, post-step `exp_avg_sq` invalid | Case 3: update created the invalid state; inspect the AdamATan2 backend path. |
| State invalid at epoch start or immediately before the step | Case 4: optimizer state was already corrupt. |
| First invalid forward record is a block, head, or loss tensor | Case 5: activation/logit failure predates backward. |

## Controlled Kaggle runs

The existing `--stop-after-step` option stops at the first epoch boundary at or
after its target. `--diagnose-numerics` requires this limit. These commands
target at least 1,100 optimizer steps without entering a full 100-epoch run.
Run them on the same four-GPU Kaggle image, data directory, and working tree;
do not add `--resume`. Each uses the same seed and starts with a new model and
optimizer state in its own output directory. Logs and checkpoints from previous
runs are left in their existing directories.

### A — two layers, four L cycles, three H cycles

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-depth2-audit \
  --diagnose-numerics \
  --stop-after-step 1100
```

### B — one layer, four L cycles, three H cycles

[`configs/mamba2_mlp_depth1_100_epochs.json`](../configs/mamba2_mlp_depth1_100_epochs.json)
inherits the base preset and changes only `L_layers` from 2 to 1.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_depth1_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-depth1-audit \
  --diagnose-numerics \
  --stop-after-step 1100
```

If A reproduces the failure and B completes the bounded run, compare two
separate `L_cycles=2` runs. The prepared presets change only `L_cycles` and
`L_layers` relative to the same base. Use new run directories:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_lcycles2_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-depth2-lcycles2-audit \
  --diagnose-numerics \
  --stop-after-step 1100

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_depth1_lcycles2_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-depth1-lcycles2-audit \
  --diagnose-numerics \
  --stop-after-step 1100
```

The second comparison is conditional on the observed A/B result. A clean
1,100-step run is bounded evidence, not proof of stability over 100 epochs.

## Current status and decision

The source audit identifies a longer unnormalized residual path and different
initialization in the Mamba branch than the attention baseline. It does not
show that this path caused the optimizer-state failure. The controlled GPU
comparisons have not been run from this local environment: it has neither
`mamba-ssm` nor Kaggle GPU access. The CPU fake-mixer tests can verify the
recursion and instrumentation interface but cannot validate the real Mamba
kernel or AdamATan2 CUDA update. Local regression completed with
`.venv/Scripts/python.exe -m pytest -q`: 45 passed, 1 skipped, with one
expected warning from a non-strict pretrained-load test. `compileall`, the
training CLI `--help`, and `git diff --check` also passed.

Do not treat the two-layer architecture as qualified for the 100-epoch run
until the two-layer comparison passes the bounded GPU run with finite gradients
and optimizer state. Do not apply an architecture patch based on the current
failure report alone.
