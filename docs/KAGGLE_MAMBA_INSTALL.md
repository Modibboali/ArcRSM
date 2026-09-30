# Offline Mamba-2 wheels for Kaggle

The project keeps the existing PyTorch/CUDA stack intact. The intended optional
operator wheel family is `causal-conv1d==1.6.2.post1` and
`mamba-ssm==2.3.2.post1`, built for Linux x86_64, CPython 3.12, PyTorch 2.10,
CUDA 12.x, and PyTorch's CXX11 ABI set to `TRUE`. These are target
requirements, not evidence that the wheels have been tested in this workspace.

## Prepare and attach the wheel dataset

On a separate internet-enabled, ABI-matched Linux build environment, obtain or
build compatible wheels for both packages and their exact target tags. Check
the wheel metadata and filenames before uploading; generic `linux_x86_64`
wheel tags alone do not prove compatibility with the PyTorch/CUDA ABI. Do not
source-build these CUDA extensions inside the offline Kaggle notebook.

Upload the wheel files directly in the root of a Kaggle Dataset (for example,
`mamba-ssm-wheels`), then add that dataset to the notebook's input data. The
example below assumes Kaggle mounts it at
`/kaggle/input/mamba-ssm-wheels`. Confirm the actual mounted path and wheel
inventory first:

```bash
find /kaggle/input/mamba-ssm-wheels -maxdepth 1 -type f
```

Confirm the notebook already has the intended Python, PyTorch, CUDA, and ABI
stack with `python scripts/audit_environment.py`. Install the convolution
dependency first, then Mamba, without resolving or replacing their dependency
stack from the offline index:

```bash
WHEEL_DIR=/kaggle/input/mamba-ssm-wheels
python -m pip install --no-index --no-deps --find-links "$WHEEL_DIR" \
  causal-conv1d==1.6.2.post1
python -m pip install --no-index --no-deps --find-links "$WHEEL_DIR" \
  mamba-ssm==2.3.2.post1
```

`--no-deps` means this only installs the two named packages: the target
notebook must already contain their non-CUDA Python requirements (including
the matching PyTorch and Triton). If pip reports no compatible distribution,
do not remove the constraints or allow an internet/source fallback; correct
the attached wheelhouse for the target ABI.

## Qualify before benchmarking

Run the required environment, dependency, standalone Mamba, attention TRM, and
hybrid TRM checks first:

```bash
python scripts/kaggle_gpu_qualification.py --batch-size 1
```

Only after the eager report passes, the optional compile checks can be attempted
separately with `--compile`. A compile failure is reported independently and
does not erase the eager results. The scripts report observed results; they do
not certify an extension wheel based on its filename.

Compare operator performance on the same small batch and settings with:

```bash
python scripts/kaggle_operator_benchmark.py \
  --batch-size 1 --seq-len 900 --warmup 2 --repeats 5
```

Both scripts default to a single sample per batch; increase the batch size
deliberately after confirming memory headroom. Record the JSON lines with the
notebook output when comparing runs. The benchmark measures median wall-clock
forward and forward-plus-backward latency after warmup, plus peak allocated and
reserved memory for a representative training step.
