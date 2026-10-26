"""What utils.carry_over() puts onto a pose read back from a program.

Needs PyRosetta, since a pose and its extra scores come from it, but no GPU
and no wrapped tool.
"""

import pytest

pytest.importorskip("pyrosetta", reason="the integration tests need PyRosetta")


@pytest.fixture(scope="module")
def rosetta():
	import rosettalink

	rosettalink.init("-mute all")
	import pyrosetta

	return pyrosetta


@pytest.fixture
def labelled(rosetta):
	"""A pose carrying one reslabel per residue and two extra scores."""
	from pyrosetta.rosetta.core.pose import setPoseExtraScore

	pose = rosetta.pose_from_sequence("AAAA")
	for resnum in range(1, pose.total_residue() + 1):
		pose.pdb_info().add_reslabel(resnum, f"res{resnum}")
	setPoseExtraScore(pose, "hb1_Rank", 1.0)
	setPoseExtraScore(pose, "hb1_note", "first")
	return pose


def test_reslabels_and_scores_both_arrive(rosetta, labelled):
	from rosettalink.utils import carry_over

	fresh = rosetta.pose_from_sequence("AAAA")
	carry_over(fresh, labelled)

	assert list(fresh.pdb_info().get_reslabels(2)) == ["res2"]
	assert fresh.scores["hb1_Rank"] == 1.0
	assert fresh.scores["hb1_note"] == "first"


def test_energies_of_the_input_pose_are_not_carried(rosetta, labelled):
	from rosettalink.utils import carry_over

	rosetta.get_score_function()(labelled)
	fresh = rosetta.pose_from_sequence("AAAA")
	carry_over(fresh, labelled)

	assert "fa_atr" not in fresh.scores
	assert fresh.scores["hb1_Rank"] == 1.0


def test_resnums_map_each_position_to_the_residue_it_came_from(rosetta, labelled):
	from rosettalink.utils import carry_over

	# A prediction of the 2nd and 4th residue only, as subset_selector gives.
	fresh = rosetta.pose_from_sequence("AA")
	carry_over(fresh, labelled, [2, 4])

	assert list(fresh.pdb_info().get_reslabels(1)) == ["res2"]
	assert list(fresh.pdb_info().get_reslabels(2)) == ["res4"]


def test_a_shorter_new_pose_takes_what_fits(rosetta, labelled):
	from rosettalink.utils import carry_over

	fresh = rosetta.pose_from_sequence("AA")
	carry_over(fresh, labelled)

	assert list(fresh.pdb_info().get_reslabels(1)) == ["res1"]
	assert fresh.scores["hb1_Rank"] == 1.0


def test_the_movers_own_scores_win_when_set_after(rosetta, labelled):
	from pyrosetta.rosetta.core.pose import setPoseExtraScore

	from rosettalink.utils import carry_over

	fresh = rosetta.pose_from_sequence("AAAA")
	carry_over(fresh, labelled)
	setPoseExtraScore(fresh, "hb1_Rank", 2.0)

	assert fresh.scores["hb1_Rank"] == 2.0


def test_a_pose_with_nothing_to_give_changes_nothing(rosetta):
	from rosettalink.utils import carry_over

	empty = rosetta.pose_from_sequence("AAAA")
	fresh = rosetta.pose_from_sequence("AAAA")
	carry_over(fresh, empty)

	assert not list(fresh.pdb_info().get_reslabels(1))
