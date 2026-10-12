"""A filter between two batched stages, against stubbed programs.

RFdiffusion writes two backbones whose chain A lengths differ by one residue,
a FOR_EACH_POSE drops the longer, and LigandMPNN designs sequences for what is
left. The number of output structures says which poses reached LigandMPNN:

    survivors x NUM_SEQUENCES   every surviving backbone, and only those
    NUM_BACKBONES x NUM_SEQUENCES   the rejected backbone was designed anyway
    NUM_SEQUENCES                   only the pose passed to apply() arrived

Run it with pytest, or directly:

    PYTHONPATH=src python tests/integration/test_filter_between_stages.py
"""

import pytest

INPUT = "examples/input_data/insulin_target.pdb"
PROTOCOL = "tests/integration/protocols/filter_between_stages.xml"
CONTIG = "[B1-150/0 90-120]"
NUM_BACKBONES = 2
NUM_SEQUENCES = 2

# The binder fixture's backbones carry 99 and 100 residues on chain A, so this
# keeps the first and drops the second. Dropping the second is what tells the
# three cases above apart: a stage that ignores the filter and takes both
# backbones produces twice as many structures.
MAX_BINDER_RESIDUES = 99
SURVIVORS = 1


@pytest.fixture(scope="module")
def results(run_protocol, tmp_path_factory):
	return run_protocol(
		"binder",
		["rfdiffusion", "ligandmpnn"],
		PROTOCOL,
		[INPUT],
		{
			"contig": CONTIG,
			"num_backbones": NUM_BACKBONES,
			"num_sequences": NUM_SEQUENCES,
			"max_binder_residues": MAX_BINDER_RESIDUES,
			"work_dir": tmp_path_factory.mktemp("filter_between_stages"),
		},
	)


def test_only_the_surviving_backbone_is_designed(results):
	output_dir, rows = results

	assert len(rows) == SURVIVORS * NUM_SEQUENCES
	assert len(sorted(output_dir.glob("*.pdb"))) == SURVIVORS * NUM_SEQUENCES


def test_every_output_comes_from_a_backbone_the_filter_passed(results):
	import pyrosetta

	output_dir, rows = results

	for row in rows:
		pose = pyrosetta.pose_from_file(str(output_dir / row["description"]))
		binder = sum(
			1
			for resnum in range(1, pose.total_residue() + 1)
			if pose.pdb_info().chain(resnum) == "A"
		)
		assert binder <= MAX_BINDER_RESIDUES


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
