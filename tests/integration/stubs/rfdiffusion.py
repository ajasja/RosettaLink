"""Stands in for RFdiffusion run_inference.py.

RFdiffusion takes hydra style key=value arguments rather than flags. This
reads the ones the mover sets and writes the layout it reads back, replaying
the designs of a recorded run:

    {output_prefix}_{i}.pdb     i from 0, inference.num_designs of them
    {output_prefix}_{i}.trb

Each pdb is copied together with the .trb recorded beside it, because the
mover takes the residue sets it labels from that .trb and the two only agree
as a pair.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from replay import StubError, fixture_dir, log


def setting(argv, name, default=None):
	"""The value of a hydra key=value argument."""
	for argument in argv:
		if argument.startswith(f"{name}="):
			return argument.split("=", 1)[1]
	return default


def recorded_designs(fixture):
	"""The recorded pdb/trb pairs, sorted."""
	pairs = []
	for pdb in sorted(fixture.glob("*.pdb")):
		trb = pdb.with_suffix(".trb")
		if trb.is_file():
			pairs.append((pdb, trb))
	if not pairs:
		raise StubError(f"No pdb/trb pairs recorded in {fixture}")
	return pairs


def main(argv):
	fixture = fixture_dir("rfdiffusion")
	prefix = setting(argv, "inference.output_prefix", "output/")
	num_designs = int(setting(argv, "inference.num_designs", "1"))
	pairs = recorded_designs(fixture)

	schedules = setting(argv, "inference.schedule_directory_path")
	if schedules:
		Path(schedules).mkdir(parents=True, exist_ok=True)

	for index in range(num_designs):
		pdb, trb = pairs[index % len(pairs)]
		target = Path(f"{prefix}_{index}.pdb")
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(pdb, target)
		shutil.copy2(trb, target.with_suffix(".trb"))

	log(f"wrote {num_designs} design(s) with prefix {prefix}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main(sys.argv[1:]))
	except StubError as error:
		log(f"error: {error}")
		raise SystemExit(1)
