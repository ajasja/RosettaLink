# RosettaLink

PyRosetta movers that wrap external protein-design programs (RFDiffusion,
LigandMPNN, ColabFold, Boltz2, ESMFold2, HBDesigner) so they can be composed
as ordinary RosettaScripts tags.

## Working on this repo

```bash
pip install -e .
PYTHONPATH=src python -m pytest tests/unit -q
```

The unit tests run **without PyRosetta**, on purpose: the cluster is often
unavailable, and PyRosetta is Linux-only. Keep it that way — import
`pyrosetta` inside the function that needs it, never at module scope in
`utils.py` or `scripts/`. Mover modules may import it at module scope, since
they only load inside a PyRosetta process.

`tests/unit/test_mover_contracts.py` reads every mover with `ast` and enforces
the two rules that otherwise only fail mid-job: every option assigned in
`__init__`/`parse_my_tag` must be copied by `clone()`, and no schema string
may contain `<`, `>` or `&`. Run it after touching any mover.

Integration tests (anything needing PyRosetta or a GPU) go in
`tests/integration/` and are not part of the default run.

## Documentation already in the repo

Read these before changing a mover; they are the worked reference and are
kept current:

- `docs/writing_movers.md` — the mover contract end to end. Part 1 is any
  mover (skeleton, subprocess handling, metadata across the disk boundary,
  one-to-many output, RMSD, how to verify an unfamiliar Rosetta component
  cheaply). Part 2 is the pipeline roles and backwards compatibility.
- `docs/mover_setup.md` — how each wrapped tool is installed and reached.

## Architecture

`rosettalink.init()` calls `pyrosetta.init()`, loads the configuration file,
then imports every module named in `REGISTRATION_MODULES` in
`centralized_registration.py`. Each module's `@register_mover` pushes its
Creator into the C++ `MoverFactory`. **A mover not listed there silently does
not exist** — no error, the file is simply never imported.

| role | mover | fan-out attribute |
|---|---|---|
| backbone generation | RFDiffusion | `num_designs` |
| sequence design | LigandMPNN | `batch_size` × `number_of_batches` |
| structure prediction | ColabFold, Boltz2, ESMFold2 | —, `diffusion_samples`, `num_diffusion_samples` |
| none | HBDesigner | `top_k`, ranked best first |

ESMFold2 is the only one that runs in process; the rest are subprocesses.

## Conventions that are interfaces

Breaking one of these usually produces a silently blank column, not an error.

- **Reslabels.** `all` is stamped on every residue and is the only label
  guaranteed non-empty. `fixed_chain` / `motif` / `designed` are an exact
  partition; `inpainted` is additive. A label matching nothing is not applied
  and not logged. Never rename or repurpose one.
- **Score keys.** `prefix_name` plus the tool's own metric name. Driver
  scripts read `pose.cache` by exact key.
- **Same name, same meaning, same units.** ColabFold reports pLDDT 0–100,
  Boltz reports `complex_plddt` 0–1. Do not unify them.
- **`get_additional_output()` is pull-one-at-a-time.** Return exactly one
  pose per call, `None` once exhausted. Returning a list works from Python
  and fails inside `MultiplePoseMover` with a pybind11 cast error.
- **Never write the resolved directory back onto `self.work_dir_`.** The same
  mover instance is applied more than once by `MultiplePoseMover`. Resolve
  into a local variable every `apply()`.

## Current work (branch `federico`)

### run_command (issues #8, #11)

Every mover takes `run_command`, the whole command prefix reaching the
external program, so the mover does not care whether it is a container, a
conda environment or a binary on PATH. Resolution order:

1. `run_command` on the tag
2. the mover's section in `rosettalink.config.yaml`
3. error naming both

The config file is `rosettalink.config.yaml`, searched in the current
directory then `~/.rosettalink/`, or passed as
`rosettalink.init(config="/path/to/file")`. It loads into
`rosettalink.utils.configuration`, **mutated in place** so
`from rosettalink.utils import configuration` stays valid.

The old `rfdiffusion_path`, `ligandmpnn_path` and `cmd_header` still work and
warn. For RFDiffusion and LigandMPNN they keep the old behaviour exactly —
singularity with the run directory bound to `/output` — because `run_command`
instead passes **host paths**, so a container reached that way must be able
to see the working directory. `%run_dir%` in RFDiffusion's `extra_args` is
replaced with the directory as the command sees it — `%...%` rather than
`{...}` so the runner's `--var` check does not claim it.

### Pose buffer (issue #15)

`FOR_EACH_POSE` in RosettaCommons/rosetta#494 is **an alias for
MultiplePoseMover**, not a batching mechanism: it adds a `mover` attribute and
a `PROTOCOLS` subtag so a nested `<ROSETTASCRIPTS>` is no longer required.
Poses still flow one at a time through `get_additional_output()`.

Batching is therefore a RosettaLink-side convention:
`rosettalink.utils.pose_buffer` is a process-wide `PoseBuffer` that a mover
with `batch="true"` consumes its full input set from and publishes its
results to. `owner` is the mover that published the current contents; a mover
serves `get_additional_output()` from the buffer **only while it still owns
it**, so once a downstream mover consumes the poses the upstream mover
reports nothing more and the protocol does not fan out over work already
done.

`batch` defaults to **false**. Leave it false inside a `FOR_EACH_POSE` or
`MultiplePoseMover`, which hand one pose at a time — turning both on would
make the mover consume poses its enclosing loop is also iterating.

Only LigandMPNN actually reduces invocations (`--pdb_path_multi`) and
ColabFold (one fasta, many records). RFDiffusion reads one structure per
invocation, so its `batch` only composes the pipeline.

Clear the buffer between input structures — `rosettalink.init()` and
`rosetta_link_scripts` both do. A buffer whose first pose does not match the
pose handed in is treated as stale, warned about and ignored.

### rosetta_link_scripts (issue #12)

`src/rosettalink/scripts/run_xml.py`, installed as `rosetta_link_scripts`.
Runs any RosettaScripts XML over any number of inputs. Accepts the
`rosetta_scripts` spellings it knows (`-parser:protocol`, `-s`, `-out:prefix`,
`-nstruct`, `-scorefile_format`) alongside long ones, and **forwards
everything it does not recognise to `pyrosetta.init()`**. `--var NAME=VALUE`
fills `{placeholders}`; an unfilled one is reported by name rather than
failing later inside Rosetta.

## Unverified

Nothing below has been run against a real tool — the cluster was down. Each
is a guess about an external program's behaviour, not about Rosetta:

- LigandMPNN `--pdb_path_multi` / `--fixed_residues_multi` /
  `--redesigned_residues_multi` are assumed to take a json keyed by input pdb
  path. Confirm against the installed `run.py`.
- LigandMPNN output files are assumed to be named `{input_stem}_{n}.pdb`.
  With one input pose the mover falls back to globbing every pdb, so only the
  batched path depends on this.
- ColabFold output files are assumed to be named `{record_id}_*rank_001*`.
- Whether a container reached through `run_command` can see an arbitrary
  work_dir without an explicit bind mount.

Flag anything in this state explicitly rather than presenting it as certain.

## House rules

- Comments say **what** the code does and how to use it, never **why** it was
  written that way. No rationale, no benchmarks or measurements, no "this
  used to be X", no reference to the bug that prompted it. Prefer no comment
  over restating the code.
- Do not commit, amend, push or tag unless asked in that request. Permission
  given once does not carry over.
- Schema description strings cannot contain `<`, `>` or `&` — Rosetta embeds
  them in the generated XSD and rejects them at registration, for every
  protocol.
- Attribute values must not be empty strings in XML. `work_dir=""` produces a
  misleading "not a valid option" error. Omit the attribute instead.
