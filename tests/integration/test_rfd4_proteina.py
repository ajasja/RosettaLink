"""RFD4-Proteina against a stubbed rfd4-generate and rfd4-evaluate.

Checks what the mover is responsible for: the condition spec it writes, the
structures it dumps, the designs it finds, the reslabel it stamps and the
fan-out of one input into num_replicates poses. The stub refuses to run if the
spec is missing, names no input, or names an input that was never dumped.

The output layout the stub writes is not recorded from a real run, so these
tests do not confirm that rfd4 names its files that way.

Run it with pytest, or directly:

    PYTHONPATH=src python tests/integration/test_rfd4_proteina.py
"""

import pytest

INPUT = "examples/input_data/insulin_target.pdb"
PROTOCOL = "tests/integration/protocols/rfd4_design.xml"
SPEC_PROTOCOL = "tests/integration/protocols/rfd4_spec_file.xml"
EVALUATE_PROTOCOL = "tests/integration/protocols/rfd4_evaluate.xml"

# Hydra settings at the top level, the condition spec under designs.
CONFIG = """
num_replicates: 2
batch_size: 1
seed: 11
designs:
  from_config:
    contig: "B1-150, /0, 90-120"
    conditions:
      C_CRD:
        true:
          - select: "sel('B/*/*/[N,CA,C,O]') and not has_nan_coord()"
"""
CONTIG = "B1-150, /0, 90-120"
NUM_REPLICATES = 2

# A spec no combination of attributes would produce: its own design name, a
# hotspot condition and a Hydra override.
SPEC = """
from_file:
  contig: "B1-150, /0, 90-120"
  conditions:
    C_HOT:
      true:
        - select: "sel('B/*/60')"
global_args:
  seed: 7
"""


@pytest.fixture(scope="module")
def results(run_protocol, tmp_path_factory):
	"""The output directory, the score rows, and the work dir holding the
	condition spec and the dumped input."""
	work_dir = tmp_path_factory.mktemp("rfd4_design")
	output_dir, rows = run_protocol(
		"binder",
		["rfd4"],
		PROTOCOL,
		[INPUT],
		{
			"contig": CONTIG,
			"num_replicates": NUM_REPLICATES,
			"work_dir": work_dir,
		},
	)
	return output_dir, rows, work_dir


@pytest.fixture(scope="module")
def spec_file_results(run_protocol, tmp_path_factory):
	"""The same run, driven by a condition spec written by hand."""
	work_dir = tmp_path_factory.mktemp("rfd4_spec_file")
	spec_file = work_dir / "my_spec.yaml"
	spec_file.write_text(SPEC)
	output_dir, rows = run_protocol(
		"binder",
		["rfd4"],
		SPEC_PROTOCOL,
		[INPUT],
		{
			"spec_file": spec_file,
			"num_replicates": NUM_REPLICATES,
			"work_dir": work_dir,
		},
	)
	return output_dir, rows, work_dir


def test_a_spec_file_is_passed_through(spec_file_results):
	"""Everything the spec file says reaches rfd4-generate, and the mover
	only fills in the input it dumped."""
	import yaml

	output_dir, rows, work_dir = spec_file_results
	spec = yaml.safe_load(sorted(work_dir.rglob("spec.yaml"))[0].read_text())

	assert list(spec) == ["from_file", "global_args"]
	assert spec["global_args"] == {"seed": 7}
	assert spec["from_file"]["input"] == "input.cif"
	assert "C_HOT" in spec["from_file"]["conditions"]
	# The attribute spec would have held the kept chain; this one does not.
	assert "C_CRD" not in spec["from_file"]["conditions"]

	assert len(rows) == NUM_REPLICATES
	assert len(sorted(output_dir.glob("*.pdb"))) == NUM_REPLICATES


@pytest.fixture(scope="module")
def evaluate_results(run_protocol, tmp_path_factory):
	"""A run configured entirely from one yaml file, scored by a second
	stage of the same mover."""
	work_dir = tmp_path_factory.mktemp("rfd4_evaluate")
	config_file = work_dir / "run.yaml"
	config_file.write_text(CONFIG)
	output_dir, rows = run_protocol(
		"binder",
		["rfd4"],
		EVALUATE_PROTOCOL,
		[INPUT],
		{"config_file": config_file, "work_dir": work_dir},
	)
	return output_dir, rows, work_dir


def test_a_config_file_carries_every_setting(evaluate_results):
	"""Hydra settings written in yaml reach the command line, and the
	designs key becomes the condition spec."""
	import yaml

	_, _, work_dir = evaluate_results

	spec = yaml.safe_load(sorted(work_dir.rglob("spec.yaml"))[0].read_text())
	assert list(spec) == ["from_config"]
	assert spec["from_config"]["input"] == "input.cif"
	assert "C_CRD" in spec["from_config"]["conditions"]

	# num_replicates came from the file, with no attribute set on the tag.
	assert len(sorted((work_dir / "01_rfd4").rglob("*_generated.cif"))) == 2


def test_evaluate_scores_the_poses_it_is_given(evaluate_results):
	"""The second stage generates nothing: it returns the designs it was
	handed, carrying the metrics as score keys."""
	output_dir, rows, work_dir = evaluate_results

	assert len(rows) == 2
	assert len(sorted(output_dir.glob("*.pdb"))) == 2

	scored = [row for row in rows if "of3_oracle_ppi.overall_plddt" in row]
	assert len(scored) == 2, sorted(rows[0])
	for row in scored:
		assert row["of3_oracle_ppi.overall_iptm"] > 0
		assert "of3_refolded_backbone_rmsd.rmsd" in row
	# A path column is not a score.
	assert "of3_designed_structure_path" not in rows[0]


def test_evaluate_reads_the_structures_as_mmcif(evaluate_results):
	"""rfd4-evaluate skips a pdb it was handed and still exits 0, so the
	mover must write mmCIF."""
	_, _, work_dir = evaluate_results
	dumped = sorted((work_dir / "02_evaluate").rglob("inputs/*"))
	assert dumped, "the evaluate stage dumped no structures"
	assert all(path.suffix == ".cif" for path in dumped), dumped


def test_the_dumped_structures_carry_no_hydrogens(evaluate_results):
	"""The oracle transfers its coordinates back onto the structure it was
	given by position, so an input carrying the hydrogens a pose has fails on
	the atom count."""
	_, _, work_dir = evaluate_results
	for path in sorted((work_dir / "02_evaluate").rglob("inputs/*.cif")):
		lines = path.read_text().splitlines()
		columns = [
			line.split(".", 1)[1].strip()
			for line in lines if line.startswith("_atom_site.")
		]
		assert "type_symbol" in columns, columns
		index = columns.index("type_symbol")

		elements = [line.split()[index] for line in lines if line.startswith("ATOM")]
		assert elements, path
		# The column really is the element, so an absent H means something.
		assert "C" in elements, sorted(set(elements))
		assert "H" not in elements, f"{path} carries {elements.count('H')} hydrogens"


def test_every_replicate_is_emitted(results):
	output_dir, rows, _ = results

	assert len(rows) == NUM_REPLICATES
	assert len(sorted(output_dir.glob("*.pdb"))) == NUM_REPLICATES


def test_the_condition_spec_carries_the_contig(results):
	import yaml

	_, _, work_dir = results
	spec_files = sorted(work_dir.rglob("spec.yaml"))
	assert spec_files, "the mover wrote no condition spec"

	spec = yaml.safe_load(spec_files[0].read_text())
	designs = [key for key in spec if key != "global_args"]
	assert len(designs) == 1

	design = spec[designs[0]]
	assert design["contig"] == CONTIG
	assert design["input"] == "input.cif"


def test_the_kept_chain_is_held(results):
	"""A contig alone does not hold what it keeps: without a condition on
	them the kept coordinates and sequence are generated afresh."""
	import yaml

	_, _, work_dir = results
	spec = yaml.safe_load(sorted(work_dir.rglob("spec.yaml"))[0].read_text())
	design = spec[[key for key in spec if key != "global_args"][0]]

	conditions = design.get("conditions")
	assert conditions, "the kept chain was not held by any condition"

	coordinates = str(conditions["C_CRD"])
	sequence = str(conditions["C_SEQ"])
	assert "sel('B/" in coordinates, coordinates
	assert "has_nan_coord" in coordinates, coordinates
	assert "sel('B')" in sequence, sequence

	# The generated segment is built, so it is never held.
	assert "sel('A" not in coordinates, coordinates


def test_the_dumped_input_is_mmcif(results):
	_, _, work_dir = results
	inputs = sorted(work_dir.rglob("input.cif"))
	assert inputs, "the mover dumped no mmCIF input"
	assert "_atom_site.label_asym_id" in inputs[0].read_text()


def test_every_design_carries_the_prefixed_all_label(results):
	import pyrosetta

	output_dir, _, _ = results
	for structure in sorted(output_dir.glob("*.pdb")):
		pose = pyrosetta.pose_from_file(str(structure))
		labels = {
			label
			for resnum in range(1, pose.total_residue() + 1)
			for label in pose.pdb_info().get_reslabels(resnum)
		}
		assert "rfd4_all" in labels


def test_the_held_chain_is_recorded_in_the_structure(evaluate_results):
	"""rfd4 names the designed chain from what was conditioned, a record a
	pose does not carry, so the mover writes one for the residues
	conditioned_selector names."""
	_, _, work_dir = evaluate_results
	for path in sorted((work_dir / "02_evaluate").rglob("inputs/*.cif")):
		lines = path.read_text().splitlines()
		columns = [
			line.split(".", 1)[1].strip()
			for line in lines if line.startswith("_atom_site.")
		]
		for column in ("condition_sequence_1_residue",
				"annotation_src-coordinate-conditioned-region_1_atom",
				"condition_CoordinateGroup_1_atom"):
			assert column in columns, columns

		chain = columns.index("label_asym_id")
		atom = columns.index("label_atom_id")
		residue = columns.index("label_comp_id")
		sequence = columns.index("condition_sequence_1_residue")
		source = columns.index(
			"annotation_src-coordinate-conditioned-region_1_atom")
		coordinates = columns.index("condition_CoordinateGroup_1_atom")

		backbone, sidechain, designed = 0, 0, 0
		for line in lines:
			if not line.startswith("ATOM"):
				continue
			fields = line.split()
			if fields[chain] != "B":
				assert fields[sequence] == "<M>", fields
				assert fields[source] == ".", fields
				assert fields[coordinates] == "-1", fields
				designed += 1
			elif fields[atom] in ("N", "CA", "C", "O"):
				# The target's backbone: held sequence and coordinates both.
				assert fields[sequence] == fields[residue], fields
				assert fields[source].startswith("B"), fields
				assert fields[coordinates] == "0", fields
				backbone += 1
			else:
				# Its sidechains: the sequence is held, the coordinates are not.
				assert fields[sequence] == fields[residue], fields
				assert fields[source] == ".", fields
				assert fields[coordinates] == "-1", fields
				sidechain += 1
		assert backbone and sidechain and designed, (backbone, sidechain, designed)


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-q"]))
