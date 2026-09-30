# ARC-AGI-2 TRM research baseline

This workspace vendors the official Samsung SAIL Montreal Tiny Recursive
Models (TRM) implementation and keeps it unchanged as the control baseline for
later operator experiments.

- Upstream checkout: `external/TinyRecursiveModels/`
- Pinned revision: `c01103738605ba39d1430519b1ee0c62f4c707f8`
- License: upstream MIT license retained in the checkout
- Baseline/audit: `docs/TRM_BASELINE_AUDIT.md`
- Dependency report: `docs/DEPENDENCY_COMPATIBILITY.md`
- Kaggle 2026 filename adapter: `scripts/build_arc_agi2_from_kaggle.py`
- Environment inventory: `scripts/audit_environment.py`
- Smoke tests: `tests/`

The project-side `build_trm(..., operator="attention")` factory preserves the
upstream attention baseline. The optional `mamba2_attention` hybrid can be
architecture-tested locally with an injected fake mixer; the real CUDA-backed
Mamba dependency is lazy-loaded and must be qualified in Kaggle. The reserved
`experimental_2d_ssm` operator is explicitly unimplemented.

Run the CPU smoke suite from the repository root:

```bash
python -m pytest -q
```

For the offline Kaggle Mamba wheel procedure and manual GPU validation scripts,
see [`docs/KAGGLE_MAMBA_INSTALL.md`](docs/KAGGLE_MAMBA_INSTALL.md).

Build arrays from an attached ARC Prize 2026 dataset without copying or
renaming the competition JSON files:

```bash
python scripts/build_arc_agi2_from_kaggle.py \
  --source-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-2 \
  --output-dir /kaggle/working/data/arc2-aug-1000
```

Add `--concept-dir PATH` only when an explicitly approved offline ConceptARC
source is attached. The competition data itself does not contain ConceptARC.
