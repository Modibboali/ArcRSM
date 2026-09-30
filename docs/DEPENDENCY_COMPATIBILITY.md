# TRM dependency compatibility report

## Target and verification scope

Target supplied for Kaggle: Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8,
Triton 3.6.0, four NVIDIA L4 GPUs (SM 8.9, about 23 GiB each), with internet
disabled. The target notebook was not available in this workspace, so package
versions not stated by the target are marked **inventory required** rather
than guessed. Run `python scripts/audit_environment.py` in Kaggle to capture
them before preparing the offline wheelhouse.

Local verification used an isolated, ignored environment with Python 3.12.13,
PyTorch 2.10.0+cpu, einops 0.8.1, pydantic 2.11.7, pydantic-core 2.33.2,
argdantic 1.3.3, numba 0.61.2, and NumPy 2.2.6. It did not modify a system or
Kaggle installation.

The optional `mamba2_attention` operator additionally targets
`mamba-ssm==2.3.2.post1` and `causal-conv1d==1.6.2.post1` as Linux x86_64,
CPython 3.12 wheels built for PyTorch 2.10 / CUDA 12.x with CXX11 ABI `TRUE`.
These extensions were not installed or exercised locally. See
[`KAGGLE_MAMBA_INSTALL.md`](KAGGLE_MAMBA_INSTALL.md) for offline attachment,
install order, qualification, and benchmark commands.

## Compatibility matrix

| Package | Upstream pin | Target/current | Needed for ARC | Finding | Offline action |
|---|---:|---:|---|---|---|
| Python | README says 3.10 or similar | 3.12.13 | all paths | Local tests pass on 3.12.13. | None; use Kaggle Python. |
| torch | 2.7.0+cu126 | 2.10.0+cu128 | model, train, eval | The pin is incompatible with the requirement not to replace the Kaggle stack. Direct model/loss/backward tests pass on 2.10.0 CPU. CUDA/compile remains unverified. | Do not install or vendor the upstream torch pin. |
| triton | 3.3.0 | 3.6.0 | indirect through PyTorch compilation | Directly pinning 3.3.0 risks breaking the PyTorch 2.10 pairing. The repo never imports Triton itself. | Do not install; keep Kaggle 3.6.0. |
| mamba-ssm | not upstream | 2.3.2.post1 (intended) | `mamba2_attention` operator only | CUDA extension; neither import nor kernels have been validated against Kaggle's actual image. | Attach a target-matched wheel; install after `causal-conv1d` with `--no-index --no-deps`. |
| causal-conv1d | not upstream | 1.6.2.post1 (intended) | required by Mamba-2 | CUDA extension; target wheel compatibility remains unverified until Kaggle qualification. | Install target-matched wheel first with `--no-index --no-deps`. |
| adam-atan2 | 0.0.3 | inventory required | training only | Source-only CUDA extension. Its setup explicitly emits SM 8.9 code, so L4 is listed, but the extension must still be compiled and tested against PyTorch 2.10/CUDA 12.8. A normal isolated build can silently omit the backend and then fail at import. | Vendor the sdist plus build tools; install with `--no-build-isolation` after torch is available, or prebuild on an ABI-matched image. High-priority target test. |
| einops | 0.8.1 | inventory required | attention reshaping | Pure Python; local model tests pass. | Vendor wheel if absent. |
| pydantic / pydantic-core | 2.11.7 / 2.33.2 | inventory required | configs and metadata | CPython 3.12 compatible; local tests pass. Keep the matched pair. | Vendor wheels if absent/mismatched. |
| argdantic | 1.3.3 | inventory required | dataset-builder CLI; imported by dataset loader | Declares Python 3.12 support; local adapter tests pass. It is imported even where its CLI is unused. | Vendor wheel if absent. |
| numpy | missing from upstream requirements | inventory required | dataset, augmentation, evaluator | A real direct dependency. NumPy 2.2.6 passed locally and satisfies Numba 0.61.2's `<2.3` bound. | Add explicitly; vendor a CPython 3.12 wheel. |
| numba | 0.61.2 | inventory required | evaluator `_crop` JIT only | CPython 3.12 wheels exist; evaluator test passes with NumPy 2.2.6. | Vendor Numba and its llvmlite wheel if absent. Could later replace this small JIT only as a deliberate variant, not in the baseline. |
| tqdm | 4.67.1 | inventory required | training UI | Expected compatible; not involved in the model smoke path. | Vendor if training. |
| coolname | 2.2.0 | inventory required | automatic run names | Expected compatible; optional if every run name is provided, but `pretrain.py` imports it eagerly. | Vendor if using upstream trainer. |
| wandb | 0.21.3 | inventory required | training logging | Imported and invoked by upstream training. Offline mode/configuration is required on Kaggle. | Vendor wheel and transitives, then set W&B offline/disabled as appropriate. |
| omegaconf / hydra-core | 2.3.0 / 1.3.2 | inventory required | training configuration | Expected to run on Python 3.12 but not exercised locally in the full trainer. | Vendor wheels and test Hydra launch in Kaggle. |
| PyYAML | absent from list | inventory required | `pretrain.py` config snapshot | Direct import missing from upstream requirements (often arrives transitively). | Add and vendor explicitly. |
| huggingface-hub | 0.34.4 (listed twice) | inventory required | not used by repository source | No import found. | Do not require for ARC baseline. |
| packaging | 24.1 | inventory required | build/tooling only | No direct runtime import in repository source. | Vendor for offline source builds. |
| ninja | 1.13.0 | inventory required | CUDA-extension build acceleration | Not an ARC runtime dependency, but useful/expected when building `adam-atan2`. | Vendor for optimizer build. |
| wheel / setuptools / setuptools-scm | 0.45.1 / 80.9.0 / 9.2.0 | inventory required | packaging and optimizer build | Build-time only. Do not globally upgrade blindly. | Put compatible wheels in the wheelhouse; use only for the isolated/offline build. |
| torchvision / torchaudio | README command only | inventory required | unused | No source import. | Do not install for TRM. |

## Safe Kaggle installation policy

`requirements/kaggle-model-eval.txt` and `requirements/kaggle-training.txt`
intentionally omit torch and Triton. Prepare a wheelhouse on an internet-enabled
Linux environment, attach it as a Kaggle dataset, and install with `--no-index`.
The `adam-atan2` artifact needs special handling because PyPI provides only an
sdist and its setup conditionally skips its CUDA backend when torch is absent
from the build environment.

Conceptual offline sequence (validate against the attached wheelhouse first):

```bash
python -m pip install --no-index --find-links /kaggle/input/TRM_WHEELHOUSE \
  -r requirements/kaggle-model-eval.txt
python -m pip install --no-index --find-links /kaggle/input/TRM_WHEELHOUSE \
  --no-build-isolation adam-atan2==0.0.3
python -m pip install --no-index --find-links /kaggle/input/TRM_WHEELHOUSE \
  -r requirements/kaggle-training.txt
```

Before full training, verify that `import adam_atan2_backend` succeeds and run
one optimizer step on an L4. Neither CUDA execution, `torch.compile`, the CUDA
extension build, NCCL, nor four-GPU reduction was available for this audit.
