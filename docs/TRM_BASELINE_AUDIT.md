# Official TRM baseline and architecture audit

## Repository integration

The official archived repository is pinned as a Git submodule at
`external/TinyRecursiveModels/` and remains unmodified.

| Item | Value |
|---|---|
| Upstream | `https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git` |
| Revision | `c01103738605ba39d1430519b1ee0c62f4c707f8` |
| Commit date | `2026-03-31T20:50:39-04:00` |
| Commit subject | `Clarify update message in README` |
| License | MIT, copyright Samsung Electronics Co., Ltd.; upstream `LICENSE` retained |
| Baseline policy | No changes inside the checkout; variants live in project-owned code |

The machine-readable pin is `external/TinyRecursiveModels.upstream.json`.
Generated datasets, checkpoints, environments, caches, and submissions are
ignored at the project root.

Relevant upstream implementation files:

- `models/recursive_reasoning/trm.py`: config, carries, block, shared recursive
  module, inner recurrence, ACT wrapper, embeddings, LM/Q heads.
- `models/layers.py`: non-causal scaled-dot-product attention, RoPE, SwiGLU,
  casted projections/embeddings, RMS normalization.
- `models/losses.py`: stablemax/softmax token loss and ACT loss head.
- `models/sparse_embedding.py`: per-puzzle embedding storage and distributed
  sparse SignSGD update.
- `config/arch/trm.yaml` and `config/cfg_pretrain.yaml`: defaults.
- `pretrain.py`: model construction, deep-supervision batching, optimizer,
  checkpoint, evaluation, EMA, and distributed orchestration.
- `puzzle_dataset.py`: memory-mapped arrays, group sampling, rank sharding,
  padding, and label-mask conversion.
- `dataset/build_arc_dataset.py` and `dataset/common.py`: ARC ingestion,
  augmentation, encoding, metadata, and inverse dihedral transforms.
- `evaluators/arc.py`: token decoding, inverse augmentation, voting,
  pass@K, and submission generation.

Project-owned files intended for later modification are
`src/arc_agi_trm/` (factory and future variants), not the upstream files. The
factory exposes `operator="attention"` now and rejects every unimplemented
operator. The future hybrid should add a project-owned block/reasoning module
and select it there.

## Call graph

```text
Hydra config (cfg_pretrain + arch/trm)
  -> pretrain.launch
     -> PuzzleDataset(train/test) -> memory-mapped encoded batches
     -> create_model
        -> TinyRecursiveReasoningModel_ACTV1
        -> ACTLossHead
        -> torch.compile (unless DISABLE_COMPILE is set)
     -> train_batch / evaluate
        -> ACTLossHead.forward
           -> ACT wrapper.forward (recycle/reset per-example carry and data)
              -> Inner.forward
                 -> token + puzzle + position input encoding
                 -> repeated shared L_level calls
                    -> TinyRecursiveReasoningModel_ACTV1Block(s)
                       -> non-causal attention -> post RMSNorm
                       -> SwiGLU MLP -> post RMSNorm
                 -> lm_head(z_H) and q_head(z_H[:, 0])
           -> stablemax token loss + halting BCE
        -> evaluator.update_batch -> inverse augmentation and candidate store
     -> evaluator.result -> distributed gather -> vote/pass@K/submission
```

## Configuration, state, and exact recurrence

`TinyRecursiveReasoningModel_ACTV1Config` is a Pydantic model containing the
batch/data dimensions, recursion counts, Transformer dimensions, positional
mode, ACT settings, dtype, `mlp_t`, prefix length, and continue-loss switch.
`H_layers` is accepted but ignored. There is one shared module, `L_level`, made
from `L_layers` instances of `TinyRecursiveReasoningModel_ACTV1Block`; the same
weights update both states.

The inner carry contains `z_H` (the embedded/current proposed answer) and
`z_L` (latent reasoning), both shaped `[B, S + P, D]`. The outer carry adds
`steps: [B]`, `halted: [B]`, and `current_data`. At the start, `halted=True`, so
the first wrapper call broadcasts learned `H_init: [D]` and `L_init: [D]` over
each state and loads the first batch. On later training calls, only halted rows
accept the new dataloader row; unhalted rows retain their previous task and
detached latent carry for another supervision step.

For one inner call, with `T = H_cycles` and `n = L_cycles`:

```text
x = input_embeddings(inputs, puzzle_identifiers)
z_H, z_L = carry

no_grad, repeated T - 1 times:
    repeated n times: z_L = L_level(z_L, z_H + x)
    z_H = L_level(z_H, z_L)

with gradients, once:
    repeated n times: z_L = L_level(z_L, z_H + x)
    z_H = L_level(z_H, z_L)

next carry = detach(z_H), detach(z_L)
logits = lm_head(z_H[:, P:])
q = q_head(z_H[:, 0])
```

Thus the entire final `n + 1` shared-network calls receive gradients; this is
not HRM's last-one-step approximation. All `(T - 1)(n + 1)` earlier calls are
inside `torch.no_grad()`. With YAML defaults (`T=3`, `n=6`, two blocks), this is
21 shared-network calls and 42 block applications per supervision step: 14
blocks worth of final recursion are differentiable. The README ARC commands
override `n` to 4, producing 15 calls / 30 block applications, with the final
10 block applications differentiable.

## ARC tensor shapes

For the upstream ARC configuration:

| Boundary | Shape / dtype |
|---|---|
| Raw grid | `[rows, cols]`, each dimension 1..30, values 0..9 |
| Encoded inputs/labels on disk | `[examples, 900]`, NumPy `uint8` |
| Local batch on four GPUs (`global_batch_size=768`) | inputs/labels `[192,900]`, IDs `[192]`, collated `int32` |
| Token embedding table | `[12,512]`; PAD=0, EOS=1, colors=2..11 |
| Token embeddings | `[B,900,512]` |
| Sparse puzzle lookup | `[B,512]` (stored weights are float32, cast for forward) |
| Configured puzzle prefix | pad lookup to `[B,8192]`, reshape to `[B,16,512]` |
| Combined/scaled input embedding | `[B,916,512]`, normally bfloat16 |
| `z_H`, `z_L` | `[B,916,512]`, normally bfloat16 |
| Attention Q/K/V | each `[B,8,916,64]`; attention is non-causal |
| Shared block output | `[B,916,512]` |
| `lm_head` logits | `[B,900,12]` after removing the 16 prefix positions |
| `q_head` | `[B,2]` from `z_H[:,0]`, split to two `[B]` float32 tensors |
| Token predictions | `[B,900]` from argmax |
| Decoded candidate | rectangular `[out_rows,out_cols]`, values 0..9 |

With `puzzle_emb_ndim=512` and `puzzle_emb_len=16`, only the first prefix row
contains the looked-up vector; the remaining 15 rows are zero before positional
handling. RoPE is applied in attention, not added to those embeddings.

## ARC data flow

The verified bundled ARC-AGI-2 inputs contain 1,000 `training2` tasks, 120
`evaluation2` tasks, and 160 ConceptARC tasks. An unaugmented build completed
locally and produced 1,280 puzzle IDs plus blank, train tensors with mean
4.35390625 examples per puzzle, and 120 evaluation puzzles with mean 1.43333
test outputs.

For every raw task, the builder always routes demonstration (`train`) pairs to
the training split. Test pairs from the named evaluation subset go to the test
split; test pairs from training/ConceptARC are also usable as supervised train
examples because their solution files are present.

Each original task can produce up to 1,000 unique variants. A variant applies:

1. one of eight dihedral transforms;
2. a random permutation of colors 1..9 while keeping black 0 fixed;
3. for training examples only, a shared random top/left translation of input
   and output within the 30x30 canvas (one randomly selected example per
   augmented puzzle is left untranslated).

The builder adds 2 to color values, lays the grid in a 30x30 PAD=0 canvas,
writes EOS=1 along the bottom/right boundary where space exists, and flattens
to 900 tokens. `PuzzleDataset` changes label PAD tokens from 0 to -100.

Conceptual example path:

```text
raw task demonstrations/test grids
 -> original plus color/dihedral variants (and training translations)
 -> 30x30 flattened input/label token arrays + per-variant puzzle ID
 -> batch [B,900], [B,900], [B]
 -> [B,900,512] token embeddings + [B,16,512] puzzle prefix
 -> x, z_H, z_L all [B,916,512]
 -> shared no-grad recursions, then one full differentiable recursion
 -> logits [B,900,12] and q_halt [B]
 -> argmax [B,900]
 -> crop the maximal top-left rectangle containing only tokens 2..11
 -> subtract 2, inverse dihedral transform, inverse color permutation
 -> group candidates by original task and input-grid hashes
 -> rank by vote count first, mean sigmoid(q_halt) second
 -> emit top two candidates and compute pass@K
```

No inverse translation is implemented; this is consistent with the builder
disabling translations in the test split.

## Loss and ACT behavior

The default loss is stablemax cross-entropy computed in float64. It is averaged
over valid tokens separately for each sequence and then summed over the local
batch. Halting loss is summed binary cross-entropy between `q_halt_logits` and
whether the entire valid output sequence is exactly correct, with weight 0.5.
The training loop divides the combined loss by the global batch size before
backward.

With `no_ACT_continue=True` (the TRM default), training halts a row when
`q_halt > 0` or at `halt_max_steps=16`, subject to randomized minimum-step
exploration. Evaluation ignores learned early halting and always runs all 16
supervision steps so batch rows stay aligned. The Q head is initialized with
zero weights and bias -5. The optional continue-Q path is not part of the
baseline and appears broken: its extra inner call is unpacked as five values
although the inner model returns three.

`ACTLossHead` reports token accuracy, exact accuracy, halt accuracy, steps, and
component losses only for rows that halt on that call. It returns only requested
detached predictions to evaluators.

## Decoding, voting, and pass@K

The evaluator crops a flattened 30x30 prediction by finding the largest
top-left rectangle containing only color tokens 2..11, subtracts 2, and then
undoes the recorded dihedral/color augmentation. It hashes both the original
input and candidate output. Across ranks/augmentations, identical outputs are
aggregated as `[vote_count, mean(sigmoid(q_halt))]` and sorted lexicographically
descending. Count therefore dominates confidence. The first two unique grids
become `attempt_1` and `attempt_2`; if only one exists it is duplicated.

For each configured K, pass@K is true for an individual test input if its label
hash occurs in the first K ranked candidates. Upstream then averages test-input
correctness inside each task and averages tasks. This differs from the current
Kaggle description, which averages over all task test outputs globally; the two
metrics diverge when tasks have different numbers of test inputs. Candidate
generation/voting is preserved unchanged for the baseline.

`aggregated_voting=True` also means `begin_eval()` does not clear prior local
predictions, so repeated evaluator calls on the same object accumulate votes
across evaluation checkpoints. This behavior is preserved and should be made
an explicit experimental choice later.

## ARC Prize 2026 file-layout adapter

Upstream expects a prefix plus legacy subset names, for example:

```text
external/TinyRecursiveModels/kaggle/combined/
  arc-agi_training2_challenges.json
  arc-agi_training2_solutions.json
  arc-agi_evaluation2_challenges.json
  arc-agi_evaluation2_solutions.json
  arc-agi_concept_challenges.json
  arc-agi_concept_solutions.json
```

The current Kaggle ARC-AGI-2 competition mount exposes consolidated files with
no `2` suffix:

```text
/kaggle/input/competitions/arc-prize-2026-arc-agi-2/
  arc-agi_training_challenges.json
  arc-agi_training_solutions.json
  arc-agi_evaluation_challenges.json
  arc-agi_evaluation_solutions.json
  arc-agi_test_challenges.json
  sample_submission.json
```

Some interactive sessions expose the competition at the older short mount
`/kaggle/input/arc-prize-2026-arc-agi-2`; pass whichever directory exists to
the adapter. The hidden test challenges have no solutions and are swapped at
submission rerun. ConceptARC is not part of the competition attachment.

`scripts/build_arc_agi2_from_kaggle.py` creates temporary links mapping
`training -> training2` and `evaluation -> evaluation2`, then delegates to the
unmodified upstream builder. It never copies or rewrites raw JSON and refuses
to overwrite a non-empty output directory. The adapter deliberately does not
pretend hidden test dummy labels are ground truth; submission-time inference
will need a separate wrapper around the attached test challenges/checkpoint.

## Checkpoints and distributed execution

Checkpoint saving writes only `model.state_dict()` to `step_N`; optimizer,
scheduler, RNG, carry, epoch, and dataloader state are not saved. Loading is
hard-coded to `map_location="cuda"`, uses `assign=True`, and only resizes the
puzzle embedding for the compiled key
`_orig_mod.model.inner.puzzle_emb.weights`. Resume is therefore weight loading,
not exact training continuation, and compiled/uncompiled key layouts require
care.

Multi-GPU execution is torch.distributed data parallelism implemented manually,
not `DistributedDataParallel`:

- `torchrun` triggers NCCL initialization and one CUDA device per local rank;
- rank 0 constructs/loads configuration and checkpoint state, then broadcasts
  parameters and buffers;
- the iterable dataset shards each global batch by rank;
- every ordinary parameter gradient is summed with `dist.all_reduce`;
- sparse puzzle-embedding gradients/IDs use `all_gather_into_tensor` and a
  custom distributed SignSGD update;
- metrics are reduced, while evaluator objects are gathered through a Gloo CPU
  group.

Because the loss is divided by global batch size before summed gradients, no
additional world-size division is expected. The upstream training/evaluation
driver is CUDA-only (`.cuda()` and CUDA device contexts are hard-coded).

## Paper/README versus implementation

- The paper says AdamW; `pretrain.py` actually uses the custom AdamATan2
  optimizer for normal parameters and SignSGD for sparse puzzle embeddings.
- The paper describes a one-position per-puzzle embedding; the code stores one
  512-value vector but pads/reshapes it into a 16-position prefix by default.
- The paper's nominal TRM is `T=3,n=6` (42 block depth), while the README ARC
  commands explicitly use `T=3,n=4` (30 block depth). YAML defaults remain 6.
- “Most common answer” is refined in code to vote count with mean halt
  confidence as the tie-breaker.
- The advertised 7M parameters exclude the potentially very large sparse
  puzzle-embedding buffer, which is an `nn.Buffer` updated separately rather
  than a normal parameter.
- `H_layers` is present in configuration but ignored.
- Current Kaggle scoring is per test output globally; upstream pass@K averages
  equally over tasks after an inner per-task average.
- The README's Python 3.10 / torch 2.7 / CUDA 12.6-era setup must not be applied
  wholesale to the supplied Python 3.12 / torch 2.10 / CUDA 12.8 stack.

## Compatibility and capacity risks

PyTorch 2.10 CPU successfully executed public APIs used by the model, including
`nn.Buffer`, SDPA, RoPE, bfloat16 forward, stablemax loss, and backward. CUDA
kernel selection and `torch.compile` remain to be tested on Kaggle. L4 supports
bfloat16 and SM 8.9, but the paper's ARC runs used four 80 GiB H100s, not four
23 GiB L4s. The upstream global batch 768 means local batch 192 on four GPUs and
may not fit unchanged.

At the theoretical maximum of 1,000 unique augmentations plus each original
for 1,280 tasks, the sparse puzzle table has 1,281,281 rows. At 512 float32
values per row it occupies about 2.44 GiB per process/GPU before other model,
activation, and optimizer memory. Full augmented input/label arrays are also
large. These are capacity warnings, not authorization to change baseline batch,
representation, augmentation, or recursion in this phase.

See `docs/DEPENDENCY_COMPATIBILITY.md` for package-specific findings.

## Project-owned operator integration

The smallest seam is the implementation behind
`TinyRecursiveReasoningModel_ACTV1ReasoningModule`, specifically its list of
per-step blocks. `src/arc_agi_trm/upstream.py` replaces only that per-step layer
list for the hybrid; `Inner.forward` and the upstream checkout remain unchanged.
The project-owned factory supports:

```text
operator=attention          -> exact upstream block
operator=mamba2_attention   -> Mamba2 -> Mamba2 -> non-causal Attention -> MLP block
operator=experimental_2d_ssm -> explicitly unimplemented
```

The hybrid preserves `[B,S+P,D] -> [B,S+P,D]`, dtype/device, post-norm
residual semantics, and the upstream `cos_sin` call contract. Its Mamba import
is lazy. `FakeMamba2` is injected only in CPU architecture tests; it is not a
Mamba implementation and provides no evidence about CUDA kernels.

## Verification performed

On Windows CPU, Python 3.12.13 and PyTorch 2.10.0+cpu:

- `.venv/Scripts/python.exe -m pytest -q`: 12 tests passed in 22.79 seconds.
  Coverage includes the attention baseline, carries and exact grad/no-grad
  recursion schedule; hybrid float32/bfloat16 block shape/dtype/device and
  backward; all fake Mamba parameters receiving gradients and expected
  recursive call counts; finite loss and lm/q head dimensions; an actual
  ARC-encoded 900-token batch through the hybrid test double; dataset adapter,
  and evaluator voting/submission.
- `compileall` passed for project source, tests, and Kaggle scripts. Both Kaggle
  scripts' `--help` commands ran locally without loading Mamba.
- Real bundled ARC-AGI-2 + ConceptARC unaugmented build: passed; 1,280 tasks and
  1,281 identifiers including blank, 11,584,126 bytes of generated arrays and
  metadata. Generated data remains ignored.

Not verified: real `mamba-ssm` or `causal-conv1d` import, CUDA forward/backward,
PyTorch CUDA build 2.10.0+cu128, `torch.compile`, Mamba hybrid execution,
memory/performance, or multi-GPU behavior. Use
[`KAGGLE_MAMBA_INSTALL.md`](KAGGLE_MAMBA_INSTALL.md) and the qualification
script on the actual four-L4 Kaggle image before drawing conclusions about
real Mamba/CUDA correctness or performance. Full training and experimental
2D SSM remain out of scope.
