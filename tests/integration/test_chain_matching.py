"""Pairing the chains of a prediction with the chains of the pose.

A predictor names the chains of a homo-oligomer in whatever order it likes.
Comparing residue i of chain A against residue i of the chain of that name
then measures the naming rather than the structure, so an RMSD pairs the
chains by fit first.

Needs PyRosetta, but no GPU and no predictor: the prediction is stood in for
by the same trimer with two of its chains exchanged.
"""

import pytest

pytest.importorskip("pyrosetta", reason="the integration tests need PyRosetta")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIMER = ROOT / "examples" / "input_data" / "coiled_coil_trimer.pdb"


@pytest.fixture(scope="module")
def mover():
	import rosettalink

	rosettalink.init("-mute all")
	from rosettalink.movers.ColabFold import ColabFold

	return ColabFold()


@pytest.fixture(scope="module")
def trimer(mover):
	import pyrosetta

	return pyrosetta.pose_from_file(str(TRIMER))


def chains_of(pose):
	info = pose.pdb_info()
	seen = []
	for resnum in range(1, pose.total_residue() + 1):
		if info.chain(resnum) not in seen:
			seen.append(info.chain(resnum))
	return "".join(seen)


def reordered(pose, order, tmp_path):
	"""The same trimer with its chains written in another order, relabelled
	A, B, C as a predictor would."""
	import pyrosetta

	parts = pose.split_by_chain()
	built = pyrosetta.Pose()
	for index in order:
		pyrosetta.rosetta.core.pose.append_pose_to_pose(built, parts[index], True)
	path = tmp_path / "reordered.pdb"
	pyrosetta.dump_pdb(built, str(path))
	return pyrosetta.pose_from_file(str(path))


def test_a_prediction_with_two_chains_exchanged_is_paired_back(mover, trimer, tmp_path):
	swapped = reordered(trimer, (1, 3, 2), tmp_path)
	folded = list(range(1, trimer.total_residue() + 1))

	matched = mover._matched_chain_mapping(swapped, trimer, folded)

	assert matched is not None
	# Chain B of the prediction holds what chain C of the pose holds.
	assert matched[1] == 1
	assert matched[51] != 51


def test_the_chains_as_named_are_kept_when_they_already_fit(mover, trimer, tmp_path):
	same = reordered(trimer, (1, 2, 3), tmp_path)
	folded = list(range(1, trimer.total_residue() + 1))

	assert mover._matched_chain_mapping(same, trimer, folded) is None


def test_matching_leaves_both_poses_alone(mover, trimer, tmp_path):
	swapped = reordered(trimer, (1, 3, 2), tmp_path)
	folded = list(range(1, trimer.total_residue() + 1))
	before = chains_of(swapped), swapped.sequence(), chains_of(trimer), trimer.sequence()

	mover._matched_chain_mapping(swapped, trimer, folded)

	assert (chains_of(swapped), swapped.sequence(),
	        chains_of(trimer), trimer.sequence()) == before


def test_chains_of_unlike_length_are_never_paired(mover, trimer, tmp_path):
	import pyrosetta

	parts = trimer.split_by_chain()
	uneven = pyrosetta.Pose()
	pyrosetta.rosetta.core.pose.append_pose_to_pose(uneven, parts[1], True)
	shorter = pyrosetta.Pose(parts[2], 1, parts[2].total_residue() - 5)
	pyrosetta.rosetta.core.pose.append_pose_to_pose(uneven, shorter, True)
	path = tmp_path / "uneven.pdb"
	pyrosetta.dump_pdb(uneven, str(path))
	uneven = pyrosetta.pose_from_file(str(path))

	folded = list(range(1, uneven.total_residue() + 1))
	pairings = mover._chain_pairings(uneven, uneven, folded)

	assert len(pairings) == 1


def test_a_homotrimer_offers_every_order(mover, trimer):
	folded = list(range(1, trimer.total_residue() + 1))

	pairings = mover._chain_pairings(trimer, trimer, folded)

	assert len(pairings) == 6
	# The identity comes first, so an equal fit keeps the chains as named.
	assert pairings[0] == dict(zip(folded, folded))


def test_match_chains_is_on_unless_the_tag_turns_it_off(mover):
	from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

	protocol = """
	<ROSETTASCRIPTS>
	  <MOVERS>
	    <ColabFold name="on" msa_mode="single_sequence"/>
	    <ColabFold name="off" msa_mode="single_sequence" match_chains="false"/>
	  </MOVERS>
	  <PROTOCOLS></PROTOCOLS>
	</ROSETTASCRIPTS>
	"""
	objects = XmlObjects.create_from_string(protocol)

	assert objects.get_mover("on").match_chains_ is True
	assert objects.get_mover("off").match_chains_ is False


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
