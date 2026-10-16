"""The residue lists LigandMPNN is handed.

Residues reach LigandMPNN only through fixed_selector and design_selector;
naming them on the tag is not an option the mover declares. The selector is
read once per input pose and written out in the space separated
chain-and-number form the tool takes.

Needs PyRosetta, since the mover imports it at module scope, but no GPU and
no LigandMPNN.
"""

import pytest

pytest.importorskip("pyrosetta", reason="the integration tests need PyRosetta")


@pytest.fixture(scope="module")
def mover():
	import rosettalink

	rosettalink.init("-mute all")
	from rosettalink.movers.LigandMPNN import LigandMPNN

	return LigandMPNN


@pytest.fixture(scope="module")
def pose(mover):
	import pyrosetta

	return pyrosetta.pose_from_sequence("AAAA")


def test_a_selector_gives_the_form_ligandmpnn_takes(mover, pose):
	from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

	assert mover._residues_for_selector(pose, ChainSelector("A")) == "A1 A2 A3 A4"


def test_no_selector_picks_nothing(mover, pose):
	assert mover._residues_for_selector(pose, None) == ""


def test_the_residues_go_back_to_pose_numbering(mover, pose):
	from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector

	residues = mover._residues_for_selector(pose, ResidueIndexSelector("2,4"))

	assert residues == "A2 A4"
	assert sorted(mover._resnums_for_residues(pose, residues)) == [2, 4]


def test_a_residue_named_on_the_tag_is_refused(mover):
	from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

	for attribute in ("fixed_residues", "redesigned_residues"):
		xml = f"""
		<ROSETTASCRIPTS>
		  <MOVERS><LigandMPNN name="s" {attribute}="A12"/></MOVERS>
		  <PROTOCOLS><Add mover="s"/></PROTOCOLS>
		</ROSETTASCRIPTS>
		"""
		with pytest.raises(RuntimeError, match=f"'{attribute}' is not allowed"):
			XmlObjects.create_from_string(xml)


def test_a_selector_is_still_accepted(mover):
	from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

	XmlObjects.create_from_string("""
	<ROSETTASCRIPTS>
	  <RESIDUE_SELECTORS><Chain name="binder" chains="A"/></RESIDUE_SELECTORS>
	  <MOVERS><LigandMPNN name="s" design_selector="binder"/></MOVERS>
	  <PROTOCOLS><Add mover="s"/></PROTOCOLS>
	</ROSETTASCRIPTS>
	""")


def test_a_membrane_selector_refuses_more_than_one_pose(mover, pose):
	from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
	from pyrosetta.rosetta.utility import (
		vector1_std_shared_ptr_core_pose_Pose_t as PoseVector,
	)

	design = mover(transmembrane_buried_selector=ChainSelector("A"))

	batch = PoseVector()
	batch.append(pose.clone())
	batch.append(pose.clone())
	design.set_batched_poses(batch)

	with pytest.raises(ValueError, match="no _multi parameter"):
		design.apply(pose.clone())


def test_a_membrane_selector_is_fine_with_one_pose(mover, pose):
	from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector

	design = mover(transmembrane_buried_selector=ResidueIndexSelector("1,3"))

	# One pose reaches the selector, so nothing stops it before the run.
	assert design._residues_for_selector(pose, design.transmembrane_buried_selector_) == "A1 A3"


def test_the_membrane_residues_cannot_be_named_on_the_tag(mover):
	from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

	for attribute in ("transmembrane_buried", "transmembrane_interface"):
		xml = f"""
		<ROSETTASCRIPTS>
		  <MOVERS><LigandMPNN name="s" {attribute}="A12"/></MOVERS>
		  <PROTOCOLS><Add mover="s"/></PROTOCOLS>
		</ROSETTASCRIPTS>
		"""
		with pytest.raises(RuntimeError, match=f"'{attribute}' is not allowed"):
			XmlObjects.create_from_string(xml)


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
