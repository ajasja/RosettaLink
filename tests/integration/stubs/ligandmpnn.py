"""Stands in for LigandMPNN run.py.

Reads the options the LigandMPNN mover passes and writes the output layout it
reads back, replaying the pdbs of a recorded run:

    seqs/{stem}.fa
    backbones/{stem}_{n}.pdb          n from 1, batch_size * number_of_batches of them
    packed/{stem}_packed_{n}_1.pdb    instead, with --pack_side_chains 1

Reached through run_command, so no PATH manipulation is involved:

    LigandMPNN:
      run_command: python tests/integration/stubs/ligandmpnn.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from replay import StubError, emit, fixture_dir, log, option, recorded_files

TRUE_VALUES = {"1", "true", "True", "yes"}


def input_stems(argv):
	"""The stems of the structures to design, from --pdb_path or the json of
	--pdb_path_multi."""
	multi = option(argv, "--pdb_path_multi")
	if multi:
		with open(multi) as handle:
			return [Path(path).stem for path in json.load(handle)]

	single = option(argv, "--pdb_path")
	if not single:
		raise StubError("neither --pdb_path nor --pdb_path_multi was passed")
	return [Path(single).stem]


def main(argv):
	fixture = fixture_dir("ligandmpnn")
	out_folder = Path(option(argv, "--out_folder", "."))
	stems = input_stems(argv)
	per_input = int(option(argv, "--batch_size", "1")) * int(
		option(argv, "--number_of_batches", "1")
	)
	packing = str(option(argv, "--pack_side_chains", "")) in TRUE_VALUES

	subdirectory = "packed" if packing else "backbones"
	every = recorded_files(fixture, subdirectory, "*.pdb")

	for stem in stems:
		# The designs recorded for this same input, so a replay gives back a
		# pose of the length that input had. Inputs the recording did not
		# have fall back to whatever was recorded.
		sources = sorted(p for p in every if p.name.startswith(f"{stem}_")) or every

		if packing:
			targets = [
				out_folder / subdirectory / f"{stem}_packed_{n}_1.pdb"
				for n in range(1, per_input + 1)
			]
		else:
			targets = [
				out_folder / subdirectory / f"{stem}_{n}.pdb"
				for n in range(1, per_input + 1)
			]
		emit(sources, targets)

		sequences = out_folder / "seqs" / f"{stem}.fa"
		sequences.parent.mkdir(parents=True, exist_ok=True)
		sequences.write_text(
			"".join(f">{stem}_{n}\nSEQUENCE\n" for n in range(1, per_input + 1))
		)

	log(f"wrote {per_input} design(s) for {len(stems)} input(s) into {out_folder / subdirectory}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main(sys.argv[1:]))
	except StubError as error:
		log(f"error: {error}")
		raise SystemExit(1)
