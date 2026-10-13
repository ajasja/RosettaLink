"""examples/pipeline_batched.xml end to end, against stubbed programs.

The whole binder design pipeline: RFdiffusion builds backbones against a
fixed target chain, LigandMPNN designs sequences for every backbone in one
invocation, and ColabFold refolds every sequence in one invocation. It covers
what the single stage protocols cannot:

  - RFdiffusion fanning out to several designs of different lengths
  - the reslabel scheme the contig produces, and the selectors reading it
  - LigandMPNN batching through --pdb_path_multi, with the target held fixed
  - three RMSD metrics, one of them superimposed on a different label than it
    measures

Run it with pytest, or directly:

    PYTHONPATH=src python tests/integration/test_binder_pipeline.py
"""

import pytest

INPUT = "examples/input_data/insulin_target.pdb"
PROTOCOL = "examples/pipeline_batched.xml"
CONTIG = "[B1-150/0 90-120]"
NUM_BACKBONES = 2
NUM_SEQUENCES = 2
NUM_DESIGNS = NUM_BACKBONES * NUM_SEQUENCES

TARGET_RESIDUES = 150

EXPECTED_SCORES = (
	"AF2_plddt",
	"AF2_pae",
	"AF2_scRMSD_all",
	"AF2_scRMSD_new_backbone",
	"AF2_scRMSD_binder_on_target",
)


@pytest.fixture(scope="module")
def results(run_protocol, tmp_path_factory):
	"""Runs the pipeline once for the whole module."""
	return run_protocol(
		"binder",
		["rfdiffusion", "ligandmpnn", "colabfold"],
		PROTOCOL,
		[INPUT],
		{
			"contig": CONTIG,
			"num_backbones": NUM_BACKBONES,
			"num_sequences": NUM_SEQUENCES,
			"work_dir": tmp_path_factory.mktemp("binder_work"),
		},
	)


def labelled_sequence(path, label):
	"""The one letter sequence of the residues carrying a reslabel."""
	import pyrosetta

	pose = pyrosetta.pose_from_file(str(path))
	pdb_info = pose.pdb_info()
	return "".join(
		pose.residue(resnum).name1()
		for resnum in range(1, pose.total_residue() + 1)
		if pdb_info.res_haslabel(resnum, label)
	)


def test_every_backbone_and_sequence_reaches_an_output_structure(results):
	output_dir, rows = results

	assert len(rows) == NUM_DESIGNS
	assert len(sorted(output_dir.glob("*.pdb"))) == NUM_DESIGNS


def test_every_metric_lands_on_every_design(results):
	_, rows = results

	for row in rows:
		for key in EXPECTED_SCORES:
			assert key in row, f"{row['description']} is missing {key}"


def test_the_contig_labels_a_target_and_a_binder(results):
	output_dir, rows = results

	for row in rows:
		path = output_dir / row["description"]
		assert len(labelled_sequence(path, "fixed")) == TARGET_RESIDUES
		assert len(labelled_sequence(path, "new_backbone")) > 0


def test_the_binder_length_is_within_the_contig_range(results):
	# The contig asks for 90 to 120 de novo residues, sampled per design.
	output_dir, rows = results

	for row in rows:
		binder = labelled_sequence(output_dir / row["description"], "new_backbone")
		assert 90 <= len(binder) <= 120


def test_the_target_sequence_is_not_redesigned(results):
	# fixed_reslabel holds the target fixed, so every design carries the
	# sequence the target came in with rather than a designed one.
	import pyrosetta

	output_dir, rows = results
	target = pyrosetta.pose_from_file(INPUT).sequence()

	for row in rows:
		kept = labelled_sequence(output_dir / row["description"], "fixed")
		assert kept == target, f"{row['description']} redesigned the target"


def test_aligning_on_the_target_cannot_fit_the_binder_better(results):
	# scRMSD_designed superimposes on the binder itself, which is by
	# definition the best fit for measuring it; scRMSD_binder_on_target
	# superimposes on the target instead and so cannot come out lower.
	_, rows = results

	for row in rows:
		assert row["AF2_scRMSD_binder_on_target"] >= row["AF2_scRMSD_new_backbone"]


def test_every_design_carries_the_label_the_metrics_select_on(results):
	# Do not turn this into a check that the designs differ. The stub reuses
	# a recording whenever a protocol asks for more records than the fixture
	# holds, so output structures may legitimately be copies of each other;
	# stubs/colabfold.py and readme.md describe when. This fixture matches
	# this protocol, but the assertion must not depend on that.
	output_dir, rows = results

	for row in rows:
		assert labelled_sequence(output_dir / row["description"], "all")


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
