"""Stands in for rfd4-generate and rfd4-evaluate.

Both take hydra style key=value arguments rather than flags. This reads the
ones the mover sets, checks the condition spec it wrote, and replays a
recorded fixture into the layout the mover reads back:

    {out_dir}/inference_samples/{example_id}_{n}/{example_id}_{n}_generated.cif

A scored run writes {example_id}_{n}_of3_ppi_refolded.cif into that same
directory, which this writes too.
"""

import csv
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from replay import StubError, fixture_dir, log


def setting(argv, name, default=None):
	"""The value of a hydra key=value argument."""
	for argument in argv:
		if argument.startswith(f"{name}="):
			return argument.split("=", 1)[1]
	return default


def read_spec(path):
	"""The example id and input path of the condition spec the mover wrote."""
	spec_path = Path(path)
	if not spec_path.is_file():
		raise StubError(f"No condition spec at {spec_path}")

	spec = yaml.safe_load(spec_path.read_text())
	designs = [key for key in spec if key != "global_args"]
	if len(designs) != 1:
		raise StubError(f"Expected one design in {spec_path}, found {designs}")

	example_id = designs[0]
	design = spec[example_id]
	if "input" not in design:
		raise StubError(f"Design {example_id} names no input in {spec_path}")
	if not Path(design["input"]).is_file():
		raise StubError(f"Input {design['input']} named by {spec_path} is missing")

	return example_id, design


def recorded_designs(fixture):
	"""The recorded mmCIF structures, sorted."""
	structures = sorted(fixture.glob("*.cif"))
	if not structures:
		raise StubError(f"No cif files recorded in {fixture}")
	return structures


def evaluate(argv, inputs_dir):
	"""One metrics row per input structure, keyed by the path it was read
	from, which is how the mover maps a row back onto its pose."""
	structures = sorted(
		path for path in Path(inputs_dir).iterdir() if path.suffix == ".cif")
	if not structures:
		raise StubError(f"No cif structures in {inputs_dir}")
	if not setting(argv, "metrics_tags"):
		raise StubError("rfd4-evaluate requires metrics_tags")

	metrics_dir = Path(setting(argv, "out_dir", "output")) / "evaluation_metrics"
	metrics_dir.mkdir(parents=True, exist_ok=True)

	with open(metrics_dir / "per_example_metrics.csv", "w", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow([
			"example_id", "designed_structure_path", "chain_of_interest",
			"oracle_ppi.overall_plddt", "oracle_ppi.overall_iptm",
			"refolded_backbone_rmsd.rmsd", "ppi.pass",
		])
		for index, structure in enumerate(structures):
			writer.writerow([
				structure.stem, str(structure), "A",
				90.0 - index, 80.0 - index, 1.0 + index, 1 - index,
			])

	log(f"scored {len(structures)} structure(s) from {inputs_dir}")
	return 0


def main(argv):
	fixture = fixture_dir("rfd4")
	inputs = setting(argv, "inputs", "spec.yaml")
	if Path(inputs).is_dir():
		return evaluate(argv, inputs)

	example_id, design = read_spec(inputs)
	out_dir = Path(setting(argv, "out_dir", "output"))
	num_replicates = int(setting(argv, "num_replicates", "1"))
	structures = recorded_designs(fixture)

	samples = out_dir / "inference_samples"

	for index in range(num_replicates):
		replicate_id = f"{example_id}_{index}"
		replicate_dir = samples / replicate_id
		replicate_dir.mkdir(parents=True, exist_ok=True)

		structure = structures[index % len(structures)]
		shutil.copy2(structure, replicate_dir / f"{replicate_id}_generated.cif")
		# The refold a scored run writes beside the design, which the mover
		# must not pick up as a design of its own.
		shutil.copy2(
			structure, replicate_dir / f"{replicate_id}_of3_ppi_refolded.cif"
		)

	log(
		f"wrote {num_replicates} design(s) for {example_id} into {samples}, "
		f"contig {design.get('contig')}"
	)
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main(sys.argv[1:]))
	except StubError as error:
		log(f"error: {error}")
		raise SystemExit(1)
