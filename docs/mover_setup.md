# Setting up the wrapped tools

Each mover wraps a tool that has to be installed separately. This file covers
what each one needs, where it can live, and how the mover reaches it.

`rosettalink` below means the environment that runs PyRosetta and your
pipeline script.

## Summary

| Mover | Tool comes as | Lives in | Reached through |
|---|---|---|---|
| RFDiffusion | singularity image | its own image | `rfdiffusion_path` |
| LigandMPNN | singularity image | its own image | `ligandmpnn_path` |
| ColabFold | singularity image | its own image | `cmd_header` |
| Boltz2 | pip package (CLI) | `rosettalink`, or its own env | `cmd_header` |
| ESMFold2 | pip package (library) | **must share the PyRosetta env** | imported in process |
| HBDesigner | pip package (CLI) | **must be its own env** | `cmd_header` |

Only ESMFold2 runs inside the PyRosetta process. Everything else is a
subprocess, so version conflicts between those tools cannot reach
`rosettalink`.

---

## RFDiffusion

A singularity image, passed as `rfdiffusion_path`. The mover runs:

```
singularity run --nv -B <run_dir>:/output <image> inference.output_prefix=/output/ ... -cd /output
```

The image must therefore:

- have RFDiffusion's `run_inference.py` as its runscript, accepting hydra
  arguments (`-cd`, `inference.*`)
- contain the model weights, or have them bound in
- accept `--nv` for GPU access

Build one from the Dockerfile in the RFDiffusion repository, then convert:

```bash
docker build -t rfdiffusion .
singularity build rfdiffusion.sif docker-daemon://rfdiffusion:latest
```

```xml
<RFDiffusion name="make_backbone" contig="[100-100]" num_designs="2"
    rfdiffusion_path="/path/to/rfdiffusion.sif" />
```

Check: `singularity run --nv rfdiffusion.sif --help`

## LigandMPNN

A singularity image, passed as `ligandmpnn_path`. The mover runs:

```
singularity run --nv -B <run_dir>:/output <image> --pdb_path /output/input.pdb --out_folder /output ...
```

The image must have LigandMPNN's `run.py` as its runscript, and its model
parameters inside the image. The mover defaults to
`/app/ligandmpnn/model_params/proteinmpnn_v_48_020.pt`; override with
`checkpoint_protein_mpnn` if your image puts them elsewhere.

```xml
<LigandMPNN name="make_sequence" ligandmpnn_path="/path/to/ligandmpnn.sif"
    batch_size="3" />
```

## ColabFold

A singularity image, but reached through `cmd_header`, which is the whole
command prefix up to and including `colabfold_batch`. The mover appends
`--model-order`, `--msa-mode`, `--rank`, the input FASTA and the output
directory.

```xml
<ColabFold name="predict" cmd_header="singularity run --nv /path/to/colabfold.sif colabfold_batch" />
```

MSA modes other than `single_sequence` contact the MSA server, so compute
nodes need outbound network access or a proxy. Put any proxy setup in your
submission script, not in `cmd_header`.

Note ColabFold folds one FASTA sequence, so it does not handle multi-chain
poses as complexes. Use Boltz2 or ESMFold2 for those.

## Boltz2

A pip package with a CLI. It can share the `rosettalink` environment, since
it runs as a subprocess either way.

```bash
conda activate rosettalink
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your driver
pip install -U 'boltz[cuda]'
```

Install torch **before** boltz so the CUDA build is yours, not whatever boltz
resolves. The `[cuda]` extra is required for the fused kernels; without it
boltz runs on slower fallback paths. Those kernels are CUDA 12 only.

They are also JIT-compiled by Triton at runtime, which needs a C compiler
with libc headers:

```bash
conda install -c conda-forge gcc_linux-64 sysroot_linux-64
conda deactivate && conda activate rosettalink   # re-activate to export CC
echo $CC                                         # must be non-empty
```

`extra_args="--no_kernels"` skips the kernels entirely if you would rather
not install a compiler.

```xml
<Boltz2 name="predict" cmd_header="boltz predict" cache="/path/to/.boltz" use_msa_server="false" />
```

Weights download to `--cache` (default `~/.boltz`) on first use.

Requires python `>=3.10,<3.13`.

### In its own environment

Use this to keep boltz's torch away from the PyRosetta environment, for
instance when `rosettalink` already holds another CUDA stack. The mover
reaches it identically; only `cmd_header` changes.

```bash
conda create -n boltz python=3.12
conda activate boltz
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -U 'boltz[cuda]'
conda install -c conda-forge gcc_linux-64 sysroot_linux-64
conda deactivate && conda activate boltz
echo $CC
```

```xml
<Boltz2 name="predict" cmd_header="conda run -n boltz boltz predict" ... />
```

The C toolchain goes in whichever environment runs boltz, not in
`rosettalink`, and each python version compiles its own copy of the Triton
cache. Add `--no-capture-output` after `conda run` to see boltz's output as
it goes rather than at the end.

Separating it costs nothing in speed: boltz is a CLI, so it starts a fresh
process and reloads its weights on every design either way.

## ESMFold2

A pip package with **no CLI**, so the mover imports it and folds in process.
It must therefore share the PyRosetta environment.

Sharing the environment is what makes the weights load once. ESMFold2 folds
on a 6B parameter ESMC trunk of about 24 GB; the mover keeps the loaded model
and reuses it for every design in the process. On one run that was 44 s to
load, then 22 s for the first fold and about 3 s for each one after. Reaching
esm as a subprocess instead would start a fresh process per design and pay
the 44 s every time, so loading would dominate the run.

That environment needs python 3.12 or newer, which means it cannot also hold
Boltz2's CUDA 12 kernels - keep separate environments for the Boltz2 and
ESMFold2 pipelines.

```bash
conda create -n rosettalink-esm python=3.12
conda activate rosettalink-esm
pip install esm                                   # must resolve to 3.4.0 or newer
python -c "import esm; print(esm.__version__)"
pip install pyrosetta --find-links https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release/
pip install -e .                                  # from the RosettaLink checkout
python -c "import pyrosetta, esm; pyrosetta.init(); print('ok')"
```

esm installs a CUDA 13 stack. If `nvidia-smi` reports CUDA 12.x:

```bash
pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y cuequivariance-ops-torch-cu13 cuequivariance-ops-cu13
pip install 'cuequivariance-ops-torch-cu12>=0.8.1' 'cuequivariance-ops-cu12>=0.8.1'
```

```xml
<ESMFold2 name="predict" model="biohub/ESMFold2-Fast" />
```

`biohub/ESMFold2-Fast` folds single sequences; `biohub/ESMFold2` adds MSA
conditioning. Both load a 6B parameter ESMC trunk of about 24 GB, downloaded
on first use into `$HF_HOME` (default `~/.cache/huggingface`). Point `HF_HOME`
somewhere with space and prefetch on a login node, since compute nodes
usually have no outbound network:

```bash
export HF_HOME=/path/with/space/hf
python -c "from huggingface_hub import snapshot_download; snapshot_download('biohub/ESMFold2-Fast')"
```

Then export `HF_HOME` and `HF_HUB_OFFLINE=1` in your submission script. The
weights are loaded once per process and reused by every design.

## HBDesigner

A pip package with a CLI, pinned to python 3.10 exactly, so it needs its own
environment.

```bash
git clone https://github.com/RosettaCommons/HBDesigner.git
cd HBDesigner
conda env create -f env_gpu.yaml     # or env_cpu.yaml
conda activate hbdesigner_gpu
conda install -c conda-forge git     # its trainer imports GitPython at module level
pip install -e .                     # editable, see pitfalls
run_hbdesigner --help
```

`env_gpu.yaml` pins PyRosetta behind the west mirror, which is unreachable;
repoint it before creating the environment:

```bash
sed -i 's#https://west.rosettacommons.org/pyrosetta/quarterly/release.cxx11thread.serialization#https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release.cxx11thread.serialization/#' env_gpu.yaml
```

Model weights ship with the repository, so there is nothing to download.

```xml
<HBDesigner name="design_network" cmd_header="conda run -n hbdesigner_gpu run_hbdesigner"
    n_res="3" n_samples="200" top_k="5" restore_input_sequence="true" />
```

`cpu="true"` adds `--cpu`. With pixi instead of conda:
`cmd_header="pixi run --manifest-path=/path/to/HBDesigner/pyproject.toml run_hbdesigner"`.

---

# Pitfalls

**PyRosetta downloads fail with a read timeout.** The
`west.rosettacommons.org` mirror is unreachable. Use the east mirror:
`--find-links https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release/`

**PyRosetta downloads fail with `CERTIFICATE_VERIFY_FAILED`.**
`graylab.jhu.edu` serves an incomplete certificate chain. Either
`--trusted-host graylab.jhu.edu`, or once and for all:
`pip config set global.trusted-host graylab.jhu.edu`. Set `PIP_TRUSTED_HOST`
instead when the pip call is made by conda.

**`The NVIDIA driver on your system is too old (found version 12040)`.** A
CUDA 13 torch on a CUDA 12 driver. There is no compatibility across major
versions. Reinstall torch from the `cu128` index. Within CUDA 12 minor
versions are compatible, so a cu128 build runs on a 12.4 driver.

**`fatal error: stdlib.h: No such file or directory`.** Triton is compiling
kernels with the system gcc, which has no libc headers. Install
`gcc_linux-64` and `sysroot_linux-64` into the environment and re-activate it
so `CC` is exported. The Triton cache in `~/.triton/cache` is keyed by python
ABI, so environments on different python versions each compile their own.

**`/lib64/libc.so.6: version GLIBC_2.32 not found`.** A prebuilt wheel was
built against a newer glibc than the node has. Rebuild it locally:

```bash
conda install -c conda-forge gcc_linux-64=13 gxx_linux-64=13 sysroot_linux-64 \
                             cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 cuda-libraries-dev=12.8
export CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
pip install --no-build-isolation --no-binary=<package> <package>==<version>
```

Pin gcc to 13: nvcc rejects host compilers newer than 14. `--no-build-isolation`
is required because these packages import torch at build time. Drop
`FORCE_CUDA` and the CUDA packages for a CPU-only build. A binary built this
way targets one GPU architecture, so restrict `--partition` to nodes with
that architecture.

**`pip install esm` gives a version with no `esm.models.esmfold2`.** On python
3.10 or 3.11 pip resolves to esm 3.2.x, which imports but has no ESMFold2.
Use python 3.12 or newer and check `esm.__version__` is 3.4.0 or newer.

**HBDesigner cannot find `model_weights/design_020.yaml`.** It resolves the
weights relative to its own source directory, and `model_weights/` sits
outside the package, so a regular install leaves it behind. Use
`pip install -e .`.

**`ImportError: Bad git executable`.** HBDesigner imports GitPython at module
level. `conda install -c conda-forge git`, or
`export GIT_PYTHON_REFRESH=quiet`.

**A job runs in the wrong environment.** `sbatch` starts a fresh shell, so
whatever your submission script activates wins over your interactive
environment. Echo `which python` in the script.

**Logs are silent while a tool runs.** `conda run` buffers output until the
command exits. Use `conda run --no-capture-output`, or point `cmd_header` at
the interpreter directly.

**An attribute is reported as not existing.** Attribute values must not be
empty strings; `work_dir=""` produces a misleading error. Omit the attribute
to use its default.
