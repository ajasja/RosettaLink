"""Boltz2 and ESMFold2 fold every pose of the step before them.

Both take the poses the previous step produced, fold each as its own entry in
one invocation, and label every prediction against the pose it was folded
from. Pairing a prediction with the wrong input pose carries the wrong
reslabels forward and measures every RMSD against the wrong structure, which
is what these tests catch.

Needs PyRosetta, but neither boltz nor esm: the boltz CLI is replaced by
stubs/boltz.py and ESMFold2's in-process calls are monkeypatched.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pyrosetta", reason="the integration tests need PyRosetta")

STUBS = Path(__file__).absolute().parent / "stubs"

SEQUENCE = "AAAAAAAAAA"


@pytest.fixture(scope="module")
def rosetta():
	import rosettalink

	rosettalink.init("-mute all")


@pytest.fixture
def labelled_poses(rosetta):
	"""Two poses of equal length, each carrying a reslabel of its own."""
	import pyrosetta

	poses = []
	for label in ("first", "second"):
		pose = pyrosetta.pose_from_sequence(SEQUENCE)
		info = pose.pdb_info()
		for resnum in range(1, pose.total_residue() + 1):
			info.add_reslabel(resnum, label)
		poses.append(pose)
	return poses


@pytest.fixture
def prediction_pdb(rosetta, tmp_path):
	"""The structure every stubbed prediction returns, the same length as the
	input poses so the reslabels carry over onto it."""
	import pyrosetta

	path = tmp_path / "prediction.pdb"
	pyrosetta.pose_from_sequence(SEQUENCE).dump_pdb(str(path))
	return path


def drain(mover, pose):
	"""The primary pose followed by every additional output, in order."""
	poses = [pose]
	while True:
		extra = mover.get_additional_output()
		if extra is None:
			return poses
		poses.append(extra)


def labels_of(pose):
	info = pose.pdb_info()
	return {
		label
		for resnum in range(1, pose.total_residue() + 1)
		for label in info.get_reslabels(resnum)
	}


def test_boltz2_folds_every_input_pose(
	rosetta, labelled_poses, prediction_pdb, previous_step, tmp_path
):
	from rosettalink.movers.Boltz2 import Boltz2

	mover = Boltz2(
		cmd_header=f"{sys.executable} {STUBS / 'boltz.py'} {prediction_pdb}",
		work_dir=str(tmp_path / "boltz"),
	)
	first, second = labelled_poses
	previous = previous_step(mover, second)
	assert previous is not None

	mover.apply(first)

	results = drain(mover, first)
	assert len(results) == 2
	assert [labels_of(result) for result in results] == [{"first"}, {"second"}]


def test_esmfold2_folds_every_input_pose(
	rosetta, labelled_poses, prediction_pdb, previous_step, monkeypatch, tmp_path
):
	from rosettalink.movers import ESMFold2 as module

	text = prediction_pdb.read_text()
	monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: object())
	monkeypatch.setattr(module, "fold", lambda model, chains, options: chains)
	monkeypatch.setattr(module, "as_samples", lambda result: [result])
	monkeypatch.setattr(module, "structure_text", lambda model, sample: (text, ".pdb"))
	monkeypatch.setattr(module, "confidence_scores", lambda sample: {"plddt": 0.9})

	mover = module.ESMFold2(work_dir=str(tmp_path / "esm"))
	first, second = labelled_poses
	previous = previous_step(mover, second)
	assert previous is not None

	mover.apply(first)

	results = drain(mover, first)
	assert len(results) == 2
	assert [labels_of(result) for result in results] == [{"first"}, {"second"}]
