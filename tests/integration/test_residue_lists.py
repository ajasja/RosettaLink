"""The residue lists LigandMPNN is handed.

The attributes take residues separated by commas; LigandMPNN takes them
separated by spaces. _merge_residues converts between the two and joins the
residues named on the tag with the ones a selector picks.

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


def test_commas_become_the_spaces_ligandmpnn_takes(mover):
	assert mover._merge_residues("A12,A13,B2", "") == "A12 A13 B2"


def test_space_around_a_comma_is_ignored(mover):
	assert mover._merge_residues(" A12 , A13 ,B2 ", "") == "A12 A13 B2"


def test_empty_entries_are_dropped(mover):
	assert mover._merge_residues("A12,,A13,", "") == "A12 A13"


def test_duplicates_go_once_and_keep_their_order(mover):
	assert mover._merge_residues("A12,A13", "A13,B2") == "A12 A13 B2"


def test_nothing_in_nothing_out(mover):
	assert mover._merge_residues("", "") == ""
	assert mover._merge_residues(None, None) == ""


def test_a_space_separated_list_is_refused(mover):
	with pytest.raises(ValueError, match="separated by commas"):
		mover._merge_residues("A12 A13", "")


def test_a_selector_joins_the_residues_named_on_the_tag(mover):
	import pyrosetta
	from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

	pose = pyrosetta.pose_from_sequence("AAAA")

	selected = mover._residues_for_selector(pose, ChainSelector("A"))
	assert selected == "A1,A2,A3,A4"

	merged = mover._merge_residues("A2", selected)
	assert merged == "A2 A1 A3 A4"
	assert sorted(mover._resnums_for_residues(pose, merged)) == [1, 2, 3, 4]


def test_no_selector_picks_nothing(mover):
	import pyrosetta

	pose = pyrosetta.pose_from_sequence("AAAA")
	assert mover._residues_for_selector(pose, None) == ""


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
