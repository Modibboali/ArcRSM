# ARC-AGI-2 TRM research baseline

This workspace includes the official Samsung SAIL Montreal Tiny Recursive
Models (TRM) source as an unchanged, individually tracked snapshot for the
attention control baseline and project-owned operator experiments.

- Upstream source snapshot: `external/TinyRecursiveModels/`
- Pinned revision: `c01103738605ba39d1430519b1ee0c62f4c707f8`
- License: upstream MIT license retained in the checkout
- Baseline/audit: `docs/TRM_BASELINE_AUDIT.md`
- Dependency report: `docs/DEPENDENCY_COMPATIBILITY.md`
- Kaggle ARC-AGI-2 training/validation builder: `scripts/build_arc_agi2_from_kaggle.py`
- Environment inventory: `scripts/audit_environment.py`
- Smoke tests: `tests/`

The checked-in `external/` tree is already available after cloning. This Git
repository has a pre-existing unmatched `trm-temp` gitlink without a
`.gitmodules` entry; it is left untouched. For an offline Kaggle run, attach a
copy of the pinned source and set `TRM_UPSTREAM_ROOT` to the directory
containing `models/`.

For the Mamba-2 + SwiGLU architecture, checkpoint APIs, and training commands,
see [`docs/MAMBA2_MLP_AND_CHECKPOINTS.md`](docs/MAMBA2_MLP_AND_CHECKPOINTS.md).

For the existing Mamba-2 + attention experiment details, see
[`docs/MAMBA2_HYBRID_EXPERIMENT.md`](docs/MAMBA2_HYBRID_EXPERIMENT.md).

To verify the pinned revision if Git metadata is available:

```bash
cat external/TinyRecursiveModels.upstream.json
```

For offline Kaggle runs, attach the same pinned checkout as a Kaggle Dataset
and set `TRM_UPSTREAM_ROOT` to the mounted directory containing `models/` before
running the scripts.

The project-side `build_trm(..., operator="attention")` factory preserves the
upstream attention baseline. The `mamba2_attention` and `mamba2_mlp` operators
are project-owned comparisons; Mamba dependencies load lazily and must be
qualified on the target Kaggle CUDA image. The reserved `experimental_2d_ssm`
operator is explicitly unimplemented.

Run the CPU smoke suite from the repository root:

```bash
python -m pytest -q
```

For the offline Kaggle Mamba wheel procedure and manual GPU validation scripts,
see [`docs/KAGGLE_MAMBA_INSTALL.md`](docs/KAGGLE_MAMBA_INSTALL.md).

Build the task-level-split dataset from the public training files:

```bash
python scripts/build_arc_agi2_from_kaggle.py \
  --source-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-2 \
  --output-dir /kaggle/working/data/arc2-mamba2-aug1000 \
  --num-aug 1000 --validation-tasks 100 --seed 42
```
