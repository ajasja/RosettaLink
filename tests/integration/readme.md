# Integration tests

These run a whole protocol end to end with the external programs replaced by
stubs, so they need PyRosetta but no GPU, no container and no wrapped tool.
They are not part of the default test run:

```bash
python -m pytest tests/integration -q
```

A stub replays a directory recorded from a real run. That covers everything
RosettaLink does between one program and the next, which is where its bugs
are: the command it builds, the files it looks for afterwards, the reslabels
it stamps, the pose buffer handing a fan-out to the next stage, the score
keys, and the RMSD metrics.

It does not cover the programs themselves. A stub keeps writing the layout it
recorded even after the real tool changes, so the layout assumptions belong
in a real run as well; see *Recording a fixture*.

## How a stub is reached

Through `run_command`, exactly as a container or a conda environment is. The
test writes a configuration pointing each mover at its stub:

```yaml
LigandMPNN:
  run_command: /path/to/python tests/integration/stubs/ligandmpnn.py
```

Nothing is put on `PATH` and no mover knows it is being tested. The fixture
to replay comes from `ROSETTALINK_STUB_FIXTURE`.

## Layout

```
stubs/      one per external program, plus replay.py for what they share
fixtures/   one directory per recorded run, one subdirectory per program
```

Each stub reads the arguments the mover actually passes and writes output
named for *this* invocation, so a protocol that asks for more designs than
were recorded still fans out correctly; the recorded files are cycled.

## Recording a fixture

Run the same protocol for real with `work_dir` set, then copy the outputs of
each program into `fixtures/<name>/<program>/`:

```bash
sbatch run_protocol.sh examples/redesign_colabfold.xml \
    -s examples/input_data/insulin_target.pdb \
    -out:path output/record_redesign \
    --var num_sequences=2 --var work_dir=output/record_redesign
```

Keep only what the mover reads back:

| program | keep |
|---|---|
| rfdiffusion | `output/_*.pdb` with the `.trb` beside each, flattened into the fixture root |
| ligandmpnn | `backbones/*.pdb` (or `packed/*.pdb`), `seqs/*.fa` |
| colabfold | `*_unrelaxed_rank_001_*.pdb`, `*_scores_rank_001_*.json` |

A `.trb` only describes the `.pdb` recorded with it, so the two are always
copied as a pair.

Re-record when a protocol's inputs or its per-stage counts change, since a
fixture is replayed against the input it was recorded from.
