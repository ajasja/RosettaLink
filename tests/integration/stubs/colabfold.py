"""Stands in for colabfold_batch.

Reads the fasta the ColabFold mover wrote and emits one prediction per record
in the layout the mover reads back, replaying a recorded run:

    {record_id}_unrelaxed_rank_001_{model}_seed_000.pdb
    {record_id}_scores_rank_001_{model}_seed_000.json

The part after rank_001 names the model that won the ranking, so it differs
between a single chain run and a complex, and between records of one run. It
is taken from the recorded file rather than assumed; the mover globs on
rank_001 and does not depend on the rest.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from replay import StubError, fixture_dir, log, recorded_files


def positional(argv):
	"""The input fasta and the output directory, the two positional arguments
	colabfold_batch takes last."""
	values = []
	skip = False
	for argument in argv:
		if skip:
			skip = False
			continue
		if argument.startswith("-"):
			# Every option the mover passes takes a value.
			skip = True
			continue
		values.append(argument)
	if len(values) < 2:
		raise StubError(f"expected an input fasta and an output directory, got {values}")
	return Path(values[-2]), Path(values[-1])


def record_ids(fasta):
	if not fasta.is_file():
		raise StubError(f"input fasta not found: {fasta}")
	return [
		line[1:].split()[0]
		for line in fasta.read_text().splitlines()
		if line.startswith(">")
	]


def renamed(recorded, record_id, marker):
	"""The recorded file's name with its own record id replaced by this one,
	which keeps whichever model won the recorded ranking."""
	index = recorded.name.find(marker)
	if index < 0:
		raise StubError(f"{recorded.name} does not carry {marker}")
	return f"{record_id}{recorded.name[index:]}"


def recorded_for(recorded, record_id, index, marker):
	"""The file recorded for this record, or the next one in order when the
	recording did not have that record."""
	for path in recorded:
		if path.name[: path.name.find(marker)] == record_id:
			return path
	return recorded[index % len(recorded)]


def main(argv):
	fixture = fixture_dir("colabfold")
	fasta, out_dir = positional(argv)
	ids = record_ids(fasta)

	wanted = (
		("_unrelaxed_rank_001", "*_unrelaxed_rank_001*.pdb"),
		("_scores_rank_001", "*_scores_rank_001*.json"),
	)

	out_dir.mkdir(parents=True, exist_ok=True)
	for marker, pattern in wanted:
		recorded = recorded_files(fixture, ".", pattern)
		for index, record_id in enumerate(ids):
			source = recorded_for(recorded, record_id, index, marker)
			shutil.copy2(source, out_dir / renamed(source, record_id, marker))

	log(f"wrote {len(ids)} prediction(s) into {out_dir}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main(sys.argv[1:]))
	except StubError as error:
		log(f"error: {error}")
		raise SystemExit(1)
