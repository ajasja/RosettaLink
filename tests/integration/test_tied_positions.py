"""Tied positions, which LigandMPNN calls symmetry.

A TiedPositions subtag names residue selectors tied position by position: the
first residue each one picks forms one group, the second another. A group is
held to a single amino acid, so a homodimer gives one group per position
rather than one group per chain, which is the shape LigandMPNN builds for
homo_oligomer.

Needs PyRosetta, since the mover imports it at module scope, but no GPU and
no LigandMPNN.
"""

import pytest

pytest.importorskip("pyrosetta", reason="the integration tests need PyRosetta")


PROTOCOL = """
<ROSETTASCRIPTS>
  <RESIDUE_SELECTORS>
	<Chain name="chain_A" chains="A"/>
	<Chain name="chain_B" chains="B"/>
	<Index name="one_residue" resnums="3"/>
  </RESIDUE_SELECTORS>
  <MOVERS>
	<LigandMPNN name="design" batch_size="2"{attributes}>
	  {subtag}
	</LigandMPNN>
  </MOVERS>
  <PROTOCOLS><Add mover="design"/></PROTOCOLS>
</ROSETTASCRIPTS>
"""


@pytest.fixture(scope="module")
def mover():
	import rosettalink

	rosettalink.init("-mute all")
	from rosettalink.movers.LigandMPNN import LigandMPNN

	return LigandMPNN


@pytest.fixture(scope="module")
def dimer(mover):
	"""Two chains, A1 A2 and B1 B2."""
	import pyrosetta

	pose = pyrosetta.pose_from_sequence("AA")
	pose.append_pose_by_jump(pyrosetta.pose_from_sequence("AA"), 1)
	info = pose.pdb_info()
	for resnum in range(1, pose.total_residue() + 1):
		info.chain(resnum, "A" if resnum <= 2 else "B")
		info.number(resnum, resnum if resnum <= 2 else resnum - 2)
	pose.update_pose_chains_from_pdb_chains()
	return pose


def build(subtag, attributes=""):
	from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

	xml = PROTOCOL.format(subtag=subtag, attributes=attributes)
	return XmlObjects.create_from_string(xml).get_mover("design")


def test_two_selectors_give_one_group_per_position(mover, dimer):
	design = build('<TiedPositions residues_selectors="chain_A,chain_B" weight="1"/>')

	residues, weights = mover._tied_positions_arguments(dimer, design.tied_positions_)

	assert residues == "A1,B1|A2,B2"
	assert weights == "1,1|1,1"


def test_the_weight_covers_every_tied_residue(mover, dimer):
	design = build('<TiedPositions residues_selectors="chain_A,chain_B" weight="0.25"/>')

	_, weights = mover._tied_positions_arguments(dimer, design.tied_positions_)

	assert weights == "0.25,0.25|0.25,0.25"


def test_without_a_weight_each_selector_carries_its_share(mover, dimer):
	design = build('<TiedPositions residues_selectors="chain_A,chain_B"/>')

	_, weights = mover._tied_positions_arguments(dimer, design.tied_positions_)

	assert weights == "0.5,0.5|0.5,0.5"


def test_selectors_picking_different_counts_are_refused(mover, dimer):
	design = build('<TiedPositions residues_selectors="chain_A,one_residue"/>')

	with pytest.raises(ValueError, match="same number of residues"):
		mover._tied_positions_arguments(dimer, design.tied_positions_)


def test_one_selector_is_not_a_tie(mover):
	with pytest.raises(ValueError, match="at least two"):
		build('<TiedPositions residues_selectors="chain_A"/>')


def test_the_symmetry_strings_cannot_be_named_on_the_tag(mover):
	from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

	for attribute in ("symmetry_residues", "symmetry_weights"):
		xml = f"""
		<ROSETTASCRIPTS>
		  <MOVERS><LigandMPNN name="d" {attribute}="A1,A2"/></MOVERS>
		  <PROTOCOLS><Add mover="d"/></PROTOCOLS>
		</ROSETTASCRIPTS>
		"""
		with pytest.raises(RuntimeError, match=f"'{attribute}' is not allowed"):
			XmlObjects.create_from_string(xml)


def test_tied_positions_refuses_more_than_one_pose(mover, dimer):
	from pyrosetta.rosetta.utility import (
		vector1_std_shared_ptr_core_pose_Pose_t as PoseVector,
	)

	design = build('<TiedPositions residues_selectors="chain_A,chain_B"/>')

	batch = PoseVector()
	batch.append(dimer.clone())
	batch.append(dimer.clone())
	design.set_batched_poses(batch)

	with pytest.raises(ValueError, match="no _multi parameter for symmetry_residues"):
		design.apply(dimer.clone())


def test_tied_positions_and_homo_oligomer_are_refused_together(mover, dimer):
	design = build(
		'<TiedPositions residues_selectors="chain_A,chain_B"/>',
		attributes=' homo_oligomer="1"',
	)

	with pytest.raises(ValueError, match="homo_oligomer"):
		design.apply(dimer.clone())


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
