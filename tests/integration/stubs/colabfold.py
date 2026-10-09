"""Stands in for colabfold_batch.

Reads the fasta the ColabFold mover wrote and emits one prediction per record
in the layout the mover reads back, replaying a recorded run:

    {record_id}_unrelaxed_rank_001_{model}_seed_000.pdb
    {record_id}_scores_rank_001_{model}_seed_000.json

The part after rank_001 names the model that won the ranking, so it differs
between a single chain run and a complex, and between records of one run. It
is taken from the recorded file rather than assumed; the mover globs on
rank_001 and does not depend on the rest.

Replaying a fan out wider than the recording
--------------------------------------------
WARNING: this stub reuses recordings rather than failing when a protocol asks
for more records than were recorded, and a run that leans on that is weaker
than it looks.

A fixture is recorded from one protocol at that protocol's per stage counts,
and a record id is only a position in the batch: design_2 of a four pose run
and design_2 of a six pose run are different designs, often of different
length. So a reused recording is picked by residue count, not by name. It has
to match: the mover measures its RMSDs between the prediction and the pose it
was folded from, and Rosetta rejects the pair outright when the two differ in
length.

What that costs a test:

  - some output structures are byte for byte copies of each other, so nothing
    may treat the outputs as independent designs, or count distinct values
  - every reuse prints "replaying X for Y" on stderr; a clean run prints none

Record a fixture at the protocol's own counts to get one distinct structure
per design, which also makes this fallback inert. See readme.md.
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
	records = []
	for line in fasta.read_text().splitlines():
		line = line.strip()
		if line.startswith(">"):
			records.append([line[1:].split()[0], 0])
		elif line and records:
			records[-1][1] += len(line.replace(":", ""))
	return [(name, length) for name, length in records]


def renamed(recorded, record_id, marker):
	"""The recorded file's name with its own record id replaced by this one,
	which keeps whichever model won the recorded ranking."""
	index = recorded.name.find(marker)
	if index < 0:
		raise StubError(f"{recorded.name} does not carry {marker}")
	return f"{record_id}{recorded.name[index:]}"


def residue_count(pdb):
	"""Residues of a recorded prediction, counted from its alpha carbons."""
	return sum(
		1
		for line in pdb.read_text().splitlines()
		if line.startswith("ATOM") and line[12:16].strip() == "CA"
	)


def choose(recorded, record_id, length, index):
	"""The recorded record to replay for one requested record.

	Preference goes to the record of the same name, but only when its length
	also matches, since a name is just a position in the batch. Otherwise any
	recording of the right length is used, and failing that the recordings
	are cycled.

	Everything below the first case is the reuse the module docstring warns
	about: it returns a recording belonging to another design."""
	if recorded.get(record_id) == length:
		return record_id
	same_length = [name for name, size in recorded.items() if size == length]
	if same_length:
		return same_length[index % len(same_length)]
	if record_id in recorded:
		return record_id
	return list(recorded)[index % len(recorded)]


def main(argv):
	fixture = fixture_dir("colabfold")
	fasta, out_dir = positional(argv)
	records = record_ids(fasta)

	PDB, SCORES = "_unrelaxed_rank_001", "_scores_rank_001"
	predictions = {
		path.name[: path.name.find(PDB)]: path
		for path in recorded_files(fixture, ".", f"*{PDB}*.pdb")
	}
	scores = {
		path.name[: path.name.find(SCORES)]: path
		for path in recorded_files(fixture, ".", f"*{SCORES}*.json")
	}
	lengths = {name: residue_count(path) for name, path in predictions.items()}

	out_dir.mkdir(parents=True, exist_ok=True)
	reused = 0
	for index, (record_id, length) in enumerate(records):
		source = choose(lengths, record_id, length, index)
		if source != record_id:
			reused += 1
			log(
				f"warning: no recording for {record_id} ({length} residues), replaying "
				f"{source} ({lengths[source]} residues) in its place"
			)
		shutil.copy2(predictions[source], out_dir / renamed(predictions[source], record_id, PDB))
		if source in scores:
			shutil.copy2(scores[source], out_dir / renamed(scores[source], record_id, SCORES))

	if reused:
		log(
			f"warning: {reused} of {len(records)} prediction(s) are copies of another "
			f"design, so these outputs are not independent structures. Record a "
			f"fixture at this protocol's counts to get one recording per design"
		)
	log(f"wrote {len(records)} prediction(s) into {out_dir}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main(sys.argv[1:]))
	except StubError as error:
		log(f"error: {error}")
		raise SystemExit(1)
