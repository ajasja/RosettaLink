# Examples

Every example is a RosettaScripts XML run through `rosetta_link_scripts`.

## Set up once per machine

Copy `rosettalink.config.yaml` to the directory you run from, or to
`~/.rosettalink/`, and set each `run_command` for your installation. Without
it every mover needs `run_command` on its tag.

If `rosetta_link_scripts` is not on your PATH, install the package into the
environment (`pip install -e .`) or call the module directly; the two are
interchangeable everywhere below:

```bash
python -m rosettalink.scripts.run_xml --protocol ...
```

## Run one

The four design examples take no arguments beyond an input structure:

```bash
rosetta_link_scripts \
    -parser:protocol examples/binder_design.xml \
    -s examples/input_data/insulin_target.pdb \
    -out:path output/binder
```

| protocol | input | what it does |
|---|---|---|
| `unconditional_design.xml` | none | 100 residue backbones from nothing |
| `binder_design.xml` | `insulin_target.pdb` | de novo binder against a fixed target chain |
| `motif_scaffolding.xml` | `5TPN.pdb` | new scaffold around the RSV F site II helix |
| `sequence_redesign.xml` | `redesign_target.pdb` | redesign one chain of an existing complex |
| `general_example_with_per_pose_scoring.xml` | `insulin_target.pdb` | the full pipeline, scoring every design |
| `colabfold_predict.xml` | any structure | refold one structure, no design |
| `redesign_colabfold.xml` | any structure | redesign a sequence and refold it |

`unconditional_design.xml` takes no `-s` at all, since its contig names only
de novo residues:

```bash
rosetta_link_scripts -parser:protocol examples/unconditional_design.xml \
    -out:path output/unconditional
```

Each design example writes four structures and a `score.json` of the metrics
cached on each.

## The generic one

`general_example_with_per_pose_scoring.xml` is the one to copy for new work. It
is the only example that takes `-var`, so the contig and the per stage counts
come from the command line, and it is the only one that scores every design
rather than only the first:

```bash
rosetta_link_scripts \
    -parser:protocol examples/general_example_with_per_pose_scoring.xml \
    -s examples/input_data/insulin_target.pdb \
    -out:path output/per_pose \
    -var contig="[B1-150/0 90-120]" \
    -var num_backbones=2 \
    -var num_sequences=3
```

`num_backbones` x `num_sequences` structures come out, 6 above. Quote the
contig: it contains brackets and a space.

A filter placed in the outer `PROTOCOLS` list runs on the primary pose only,
so its column is blank for every design but the first. This protocol wraps
the filters and the simple metric in a `MultiplePoseMover` with a nested
`ROSETTASCRIPTS`, which applies them to each design. That nested script has
its own parse scope, so the score function, selectors, filters and metrics
inside it are declared a second time; change one copy and change the other.

The predictor metrics need no wrapper. ColabFold attaches pLDDT, PAE and the
`RMSD` tags to every pose it produces, which is why the other examples report
them on all four designs without one.

## On the cluster

`run_protocol.sh` submits any protocol to slurm, passing everything after the
XML straight through:

```bash
mkdir -p ~/runs/binder/logs && cd ~/runs/binder
sbatch ~/RosettaLink/run_protocol.sh \
    ~/RosettaLink/examples/binder_design.xml \
    -s ~/RosettaLink/examples/input_data/insulin_target.pdb \
    -out:path out
```

Two things to get right. Make `logs/` before submitting, since slurm resolves
its output path at submit time and the job dies silently without it. And run
from a scratch directory rather than the checkout: `-out:path` and any
`work_dir` in a protocol are relative to wherever `sbatch` was invoked, so
launching from the repository writes output into it.

## Options

`rosetta_link_scripts` takes the `rosetta_scripts` spellings of the options it
knows as well as long ones, and passes anything it does not recognise to
`pyrosetta.init()`:

```bash
rosetta_link_scripts --protocol design.xml --input 1ubq.pdb --nstruct 4
rosetta_link_scripts -parser:protocol design.xml -s 1ubq.pdb -mute all
```

`-var NAME=VALUE`, `-parser:script_vars NAME=VALUE` and `--var NAME=VALUE` all
fill a `{NAME}` placeholder. An unfilled one is reported by name before
Rosetta is reached.

`--scorefile-format csv` writes a CSV instead of JSON.

## Writing your own

Start from `general_example_with_per_pose_scoring.xml`.

Each mover stamps reslabels that later stages select on, behind whatever
`prefix_name` it was given. RFdiffusion stamps `new_backbone`,
`hidden_sidechains`, `motif`, `fixed` and `all`; LigandMPNN stamps
`new_sidechains` on every residue it was free to redesign. A label matching
nothing is skipped and warned about, never an error, so a metric selecting on
an absent label simply leaves its column blank. That is why
`motif_scaffolding.xml` reports `scRMSD_motif` and `binder_design.xml` does
not: that contig places no motif.

A structure read from a pdb carries no reslabels at all, so a protocol that
starts from one stamps its own with `AddResidueLabel` before anything selects
on them; `sequence_redesign.xml` shows that.

`batch="true"` hands every pose from one stage to the next through the
RosettaLink pose buffer, so each external program runs once per stage instead
of once per pose. Leave it false inside a `MultiplePoseMover`, which already
hands over one pose at a time.

`docs/two_helices.xml` is the same pipeline written with `FOR_EACH_POSE`, and
`docs/for_each_pose_check.xml` tells you in seconds whether a PyRosetta build
supports that syntax. No build we can reach does yet.
