# Examples

## Configuration

Copy `rosettalink.config.yaml` to the directory you run from, or to
`~/.rosettalink/`, and set each `run_command` for your installation. Every
mover takes `run_command` on the tag as well, which wins over the file.

## Running an XML

`rosetta_link_scripts` runs any RosettaScripts XML over any number of input
structures:

```bash
rosetta_link_scripts \
    --protocol examples/pipeline_batched.xml \
    --input examples/input_data/insulin_target.pdb \
    --var contig="[B1-150/0 90-120]" \
    --var num_backbones=2 \
    --var num_sequences=2 \
    --var work_dir=output \
    --output-dir output \
    --scorefile-format csv
```

It also takes the `rosetta_scripts` spellings of the options it knows, and
passes anything it does not recognise to `pyrosetta.init()`:

```bash
rosetta_link_scripts -parser:protocol design.xml -s 1ubq.pdb -nstruct 4 -mute all
```

`--var NAME=VALUE` fills a `{NAME}` placeholder in the XML. An unfilled one is
reported by name before Rosetta is reached.

Every structure the protocol produces is written out, the primary pose and
everything reached through `get_additional_output()`, with a score file of the
values cached on each.

## The XML files

- `colabfold_predict.xml` - refold one structure and report its confidence
  and a self-consistency RMSD. One stage, no fan-out; the quickest check
  that an installation works.
- `redesign_colabfold.xml` - LigandMPNN redesigns the sequence of an input
  backbone and ColabFold refolds every design. Two batched stages.
- `pipeline_batched.xml` — binder design with `batch="true"` on every stage,
  so each external program runs once for the whole fan-out rather than once
  per pose. Works in a plain `PROTOCOLS` list.
- `../example.xml` (repo root) — work in progress sketch of the target
  syntax. Uses `FOR_EACH_POSE`, which needs a Rosetta build carrying
  RosettaCommons/rosetta#494, and attribute names not yet implemented
  (`prefix_name` on RFdiffusion and LigandMPNN, `residues_to_design`,
  `num_designs` on LigandMPNN, `<RMSD input= alignment=>`, nested
  `<metric>`).

An input read straight from a pdb carries no reslabels, so the single stage
and redesign protocols stamp `all` with `AddResidueLabel` for the `<RMSD>`
tag to select on. A pose coming from RFdiffusion or LigandMPNN is labelled
already.

Filters placed after a batched mover run on the primary pose only, so their
columns are blank for every other output structure.

## The python demos

The `demo_*.py` scripts predate `rosetta_link_scripts` and drive the movers
directly from python, with the XML inline as a template. New work belongs in
an XML run through `rosetta_link_scripts`.

The ones left are those that still parse. `demo_boltz2.py`,
`demo_esmfold2.py` and `demo_hbdesigner.py` are the only examples of those
three movers, which have no XML yet.
