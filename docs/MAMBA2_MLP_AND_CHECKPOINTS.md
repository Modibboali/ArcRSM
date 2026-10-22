# Mamba-2 + SwiGLU TRM and checkpoint workflows

## Audit

The repository vendors the upstream TRM under `external/TinyRecursiveModels/`;
that implementation remains unchanged. `build_trm` changes only the shared
`L_level.layers` list, leaving the outer/inner cycles, carry resets and detach,
ACT supervision, output heads, ARC vocabulary, and the upstream no-gradient
cycle behavior intact.

Before this change, `mamba2_attention` already installed two Mamba-2 mixers,
non-causal attention, and the vendored SwiGLU. Its training script already
saved atomic `latest.pt` and `best.pt` full checkpoints with both optimizers,
step-derived scheduler state, per-rank RNG/carry/sampler state, and strict
configuration fingerprints. It did not have a reusable model-only artifact or
a way to select Mamba-2 plus SwiGLU without attention.

## `mamba2_mlp` architecture

Each shared block now applies the following pre-normalized residual sequence:

```text
h1 = h + Mamba2(RMSNorm(h))
h2 = h1 + SwiGLU(RMSNorm(h1))
```

It uses one Mamba-2 module and the existing `models.layers.SwiGLU`. The block
does not instantiate attention. The Mamba scan reads the flattened token stream
in its existing order; it is not a bidirectional or two-dimensional ARC scan.
Puzzle prefix embeddings and the TRM output heads are preserved. The preset
uses `pos_encodings: "none"`; Mamba supplies sequence-order processing, and no
RoPE is applied by this block.

Select the architecture with `operator: "mamba2_mlp"`. The new presets are
`configs/mamba2_mlp_100_epochs.json` and
`configs/mamba2_mlp_1000_epochs.json`, layered over
`configs/mamba2_mlp_base.json`. The factory requires each qualified `mamba_*`
setting to be present; missing settings and unavailable Mamba dependencies
raise actionable errors and never fall back to attention.

The current Kaggle target is Python 3.12 with the dependency set documented in
[`DEPENDENCY_COMPATIBILITY.md`](DEPENDENCY_COMPATIBILITY.md): Mamba-2 uses
`mamba-ssm==2.3.2.post1` and `causal-conv1d==1.6.2.post1`, matched to the
provided PyTorch 2.10.0+cu128/CUDA 12.8 image. The offline install and
qualification procedure is in [`KAGGLE_MAMBA_INSTALL.md`](KAGGLE_MAMBA_INSTALL.md).
The real CUDA kernels still need qualification on the target Kaggle image.

## Train

Build the ARC-AGI-2 training/validation dataset using the commands in
[`MAMBA2_HYBRID_EXPERIMENT.md`](MAMBA2_HYBRID_EXPERIMENT.md), then start the
100-epoch Mamba-2 + MLP run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-100
```

Use `configs/mamba2_mlp_1000_epochs.json` and a separate run directory for the
longer experiment. The training script does not launch either run during
installation or testing. It records the architecture, all model settings,
dataset manifest/split, augmentation, optimizers, schedule, batch sizes,
precision, seed, and output paths in `config_snapshot.json` and the epoch logs.

## Save and load model-only weights

Training maintains three distinct artifacts in the run directory:

- `latest.pt`: latest full training state for resume.
- `best.pt`: full state from the best validation result. Selection maximizes
  task-level exact match, breaking ties with lower validation loss.
- `best_pretrained.pt` and `final_pretrained.pt`: model-only, tensor-safe
  artifacts for evaluation or weights-only initialization.

To export model-only weights from a trusted full checkpoint:

```bash
python scripts/export_pretrained.py \
  --checkpoint /kaggle/working/outputs/mamba2-mlp-100/best.pt \
  --output /kaggle/working/outputs/mamba2-mlp-100/exported_best.pt \
  --trusted-full-checkpoint \
  --device cpu
```

The Python API creates parent directories, writes atomically, saves the
architecture/configuration/vocabulary/schema/training metadata, and loads with
`weights_only=True`:

```python
from arc_agi_trm import load_pretrained_model

model = load_pretrained_model(
    "/kaggle/working/outputs/mamba2-mlp-100/best_pretrained.pt",
    device="cuda",
    batch_size=8,
)
model.eval()
```

The loaded object uses the existing ACT/TRM interface. Inputs are the existing
encoded ARC tokens plus `puzzle_identifiers`; pass a label tensor filled with
the ignore label for inference, then call `initial_carry` and iterate until
`all_finished`. The returned `preds` use the unchanged 12-token ARC vocabulary
and decoder. The project evaluation script below exercises that interface.
CPU loading works for all model-only artifacts. CPU inference is available for
operators whose installed dependencies support it; the qualified Mamba-2 CUDA
path reports an explicit error on CPU instead of substituting attention.

Start a new run from these weights with fresh optimizers and scheduler state:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-finetune \
  --init-from /kaggle/working/outputs/mamba2-mlp-100/best_pretrained.pt
```

`--init-from` loads only model parameters and requires matching architecture
and model settings. It constructs new optimizer/scheduler state. Strict loading
reports any architecture, configuration, key, or tensor-shape mismatch.

## Resume full training state

Resume is a separate operation. It restores model and optimizer states, the
step-derived scheduler position, completed epoch/global step, per-rank Python,
NumPy, CPU/CUDA PyTorch RNG states, sampler position, carry, and best-validation
metadata. It starts at the next epoch and requires the saved configuration,
dataset manifest, and distributed world size to match:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --run-dir /kaggle/working/outputs/mamba2-mlp-100 \
  --resume latest
```

Full checkpoints contain Python/NumPy RNG objects and TRM carry dataclasses,
so full-state loading uses pickle. Only resume or evaluate files produced by
this project's trainer; model-only artifacts use tensor-only safe loading.
The current training loop uses bfloat16 without a gradient scaler, and the
checkpoint records that no scaler state is present.

## Evaluate

Evaluate the model-only best artifact on the configured task-level validation
split:

```bash
python scripts/evaluate_mamba2_hybrid.py \
  --config configs/mamba2_mlp_100_epochs.json \
  --dataset /kaggle/working/data/arc2-mamba2-aug1000 \
  --checkpoint /kaggle/working/outputs/mamba2-mlp-100/best_pretrained.pt \
  --device cuda --batch-size 8
```

To evaluate a trusted full checkpoint instead, add
`--trusted-full-checkpoint` and point `--checkpoint` to `best.pt`. The evaluator
validates its configuration and keeps the dataset's existing token decoding.

## Verification and performance

CPU tests inject a trainable fake mixer to check block structure, recursion,
gradients, optimizer registration, safe save/load, strict compatibility, and
inference round trips. The fake is not evidence of real Mamba kernel behavior.
Run the CUDA qualification before training:

```bash
python scripts/kaggle_gpu_qualification.py --batch-size 1 --h-cycles 3 --l-cycles 4 --layers 2
python scripts/kaggle_operator_benchmark.py --batch-size 1 --seq-len 900 --warmup 2 --repeats 5
```

These scripts measure `attention`, `mamba2_attention`, and `mamba2_mlp` at the
same input dimensions. No speed or accuracy advantage is claimed without
observed results on the target GPU. The implementation does not start a
production training run.
