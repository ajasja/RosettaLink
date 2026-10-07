"""examples/generic_script_with_perPoseScoring.xml end to end, against stubs.

The whole pipeline in the form that runs on a stock PyRosetta release, which
is the one thing the other integration tests do not cover: every stage
batched, the reslabels carrying a per mover prefix, a residue selector
choosing what LigandMPNN may redesign, the confidence metric subtags, and a
MultiplePoseMover scoring each design rather than only the first.

It runs at the counts the binder fixture was recorded at, so no recording is
reused and every design is a distinct structure.

Run it with pytest, or directly:

    PYTHONPATH=src python tests/integration/test_per_pose_scoring_pipeline.py
"""

import pytest

INPUT = "examples/input_data/insulin_target.pdb"
PROTOCOL = "examples/generic_script_with_perPoseScoring.xml"
CONTIG = "[B1-150/0 90-120]"
NUM_BACKBONES = 2
NUM_SEQUENCES = 2
NUM_DESIGNS = NUM_BACKBONES * NUM_SEQUENCES

TARGET_RESIDUES = 150

# Written by the predictor, so they land on every design on their own.
PREDICTOR_SCORES = (
	"AF2_plddt_all",
	"AF2_plddt_new_backbone",
	"AF2_pAE_all",
	"AF2_pAE_new_backbone",
	"AF2_scRMSD_all",
	"AF2_scRMSD_new_backbone",
	"AF2_scRMSD_new_backbone_on_fixed",
)

# Reached only through the MultiplePoseMover, so these are what tells a run
# that scored every design from one that scored only the first.
PER_DESIGN_SCORES = ("sap_score", "apolar_segment", "netcharge", "buried_unsat")


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
		},
	)


def labelled_resnums(path, label):
	"""Residues carrying a reslabel, in pose numbering."""
	import pyrosetta

	pose = pyrosetta.pose_from_file(str(path))
	pdb_info = pose.pdb_info()
	return [
		resnum
		for resnum in range(1, pose.total_residue() + 1)
		if pdb_info.res_haslabel(resnum, label)
	]


def test_every_backbone_and_sequence_reaches_an_output_structure(results):
	output_dir, rows = results

	assert len(rows) == NUM_DESIGNS
	assert len(sorted(output_dir.glob("*.pdb"))) == NUM_DESIGNS


def test_the_predictor_scores_land_on_every_design(results):
	_, rows = results

	for row in rows:
		for key in PREDICTOR_SCORES:
			assert key in row, f"{row['description']} is missing {key}"


def test_the_multiple_pose_mover_scores_every_design(results):
	# Without it these reach the first design only, which is the whole
	# reason the scoring stage is wrapped.
	_, rows = results

	for row in rows:
		for key in PER_DESIGN_SCORES:
			assert key in row, f"{row['description']} was never scored: no {key}"


def test_a_metric_whose_label_matches_nothing_is_skipped(results):
	# This contig keeps a whole chain and places no motif, so the motif
	# metrics have nothing to average and are left off rather than reported
	# as the mean of an empty selection.
	_, rows = results

	for row in rows:
		assert "AF2_plddt_motif" not in row
		assert "AF2_pAE_motif" not in row
		assert "AF2_scRMSD_motif" not in row


def test_each_mover_labels_behind_its_own_prefix(results):
	output_dir, rows = results

	for row in rows:
		path = output_dir / row["description"]
		assert len(labelled_resnums(path, "rfd_fixed")) == TARGET_RESIDUES
		assert labelled_resnums(path, "rfd_new_backbone")
		assert labelled_resnums(path, "mpnn_new_sidechains")
		# The unprefixed names belong to no mover here.
		assert not labelled_resnums(path, "new_backbone")
		assert not labelled_resnums(path, "fixed")


def test_the_selector_decides_what_ligandmpnn_may_redesign(results):
	# residues_to_design is new_backbone, hidden_sidechains and anything
	# within 10A of the new backbone, so the redesigned set covers the new
	# backbone and reaches into the target without covering all of it.
	output_dir, rows = results

	for row in rows:
		path = output_dir / row["description"]
		redesigned = set(labelled_resnums(path, "mpnn_new_sidechains"))
		new_backbone = set(labelled_resnums(path, "rfd_new_backbone"))
		target = set(labelled_resnums(path, "rfd_fixed"))

		assert new_backbone <= redesigned
		assert redesigned & target, "the neighbourhood reached no target residue"
		assert not target <= redesigned, "the whole target was left redesignable"


def test_aligning_on_the_target_cannot_fit_the_new_backbone_better(results):
	_, rows = results

	for row in rows:
		assert (
			row["AF2_scRMSD_new_backbone_on_fixed"] >= row["AF2_scRMSD_new_backbone"]
		)


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
