"""examples/redesign_colabfold.xml end to end, against stubbed programs.

LigandMPNN designs two sequences for the input backbone and ColabFold folds
both in one invocation, so this covers the pose buffer handing a fan out from
one batched mover to the next, the reslabels surviving both disk round trips,
and the score keys and RMSD metric landing on every output pose.

Run it with pytest, or directly:

    PYTHONPATH=src python tests/integration/test_redesign_pipeline.py
"""

import pytest

INPUT = "examples/input_data/insulin_target.pdb"
PROTOCOL = "examples/redesign_colabfold.xml"
NUM_SEQUENCES = 2

EXPECTED_SCORES = ("AF2_plddt", "AF2_pae", "AF2_scRMSD_all")


@pytest.fixture(scope="module")
def results(run_protocol, tmp_path_factory):
	"""Runs the protocol once for the whole module."""
	return run_protocol(
		"redesign",
		["ligandmpnn", "colabfold"],
		PROTOCOL,
		[INPUT],
		{
			"num_sequences": NUM_SEQUENCES,
			"work_dir": tmp_path_factory.mktemp("redesign_work"),
		},
	)


def test_one_output_structure_per_designed_sequence(results):
	output_dir, rows = results

	assert len(rows) == NUM_SEQUENCES
	assert len(sorted(output_dir.glob("*.pdb"))) == NUM_SEQUENCES


def test_every_output_structure_was_written(results):
	output_dir, rows = results

	for row in rows:
		assert (output_dir / row["description"]).is_file()


def test_the_predictor_scores_land_on_every_pose(results):
	_, rows = results

	for row in rows:
		for key in EXPECTED_SCORES:
			assert key in row, f"{row['description']} is missing {key}"
			assert isinstance(row[key], float)


def test_the_rmsd_is_a_real_measurement(results):
	# A prediction superimposed on its own backbone cannot be a negative
	# distance, and an RMSD of exactly zero would mean the metric never ran.
	_, rows = results

	for row in rows:
		assert row["AF2_scRMSD_all"] > 0


def test_reslabels_survive_both_disk_round_trips(results):
	# The label is stamped before LigandMPNN, carried onto its designs, then
	# onto the ColabFold prediction, and is what the RMSD selects on.
	output_dir, rows = results

	for row in rows:
		text = (output_dir / row["description"]).read_text()
		assert "PDBinfo-LABEL" in text
		assert " all" in text


def test_each_design_is_folded_separately(results):
	# One record per pose, so two poses must not come back as copies of one
	# prediction.
	_, rows = results

	plddts = {row["AF2_plddt"] for row in rows}
	assert len(plddts) == NUM_SEQUENCES


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
