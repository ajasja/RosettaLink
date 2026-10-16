"""Pairing the chains of a prediction with the chains of a reference.

utils.match_chains works on bare CA coordinates, so these run without
PyRosetta. The oligomers are built here: a helix copied around the z axis,
each copy given its own small deformation so that only one pairing fits.
"""

import numpy as np
import pytest

from rosettalink.utils import match_chains
from rosettalink.utils import superimposed_rmsd


def helix(length):
	k = np.arange(length)
	angle = np.radians(100.0) * k
	return np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * k], axis=1)


def ring(chains, count, radius, rng=None):
	"""`chains` placed `count` times around the z axis, cycling through them.
	With `rng`, each copy is deformed by its own noise."""
	placed = []
	for copy in range(count):
		theta = 2 * np.pi * copy / count
		rotation = np.array([
			[np.cos(theta), -np.sin(theta), 0.0],
			[np.sin(theta), np.cos(theta), 0.0],
			[0.0, 0.0, 1.0],
		])
		chain = (chains[copy % len(chains)] + [radius, 0.0, 0.0]) @ rotation.T
		if rng is not None:
			chain = chain + rng.normal(scale=0.5, size=chain.shape)
		placed.append(chain)
	return placed


def rigid_motion(rng):
	rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
	if np.linalg.det(rotation) < 0:
		rotation[:, 0] = -rotation[:, 0]
	return rotation, rng.normal(scale=20.0, size=3)


def predict(reference, order, rng):
	"""The reference with its chains reordered so that chain i of the result
	is reference chain order[i], moved as a whole and lightly perturbed."""
	rotation, translation = rigid_motion(rng)
	return [
		reference[j] @ rotation + translation + rng.normal(scale=0.2, size=reference[j].shape)
		for j in order
	]


@pytest.mark.parametrize("count", [2, 3, 5, 8, 24])
def test_a_shuffled_homo_oligomer_is_paired_back(count):
	rng = np.random.default_rng(count)
	reference = ring([helix(30)], count, radius=4.0 * count, rng=rng)
	order = rng.permutation(count).tolist()

	assert match_chains(predict(reference, order, rng), reference, ["A"] * count) == order


def test_the_chains_as_named_are_kept_when_they_already_fit():
	rng = np.random.default_rng(0)
	reference = ring([helix(30)], 4, radius=16.0, rng=rng)

	assert match_chains(predict(reference, [0, 1, 2, 3], rng), reference, ["A"] * 4) == [0, 1, 2, 3]


def test_an_exactly_symmetric_complex_against_itself_keeps_the_chains_as_named():
	symmetric = ring([helix(30)], 3, radius=12.0)

	assert match_chains(symmetric, symmetric, ["A"] * 3) == [0, 1, 2]


def test_chains_pair_only_with_chains_of_their_own_kind():
	rng = np.random.default_rng(1)
	reference = ring([helix(30), helix(20)], 4, radius=16.0, rng=rng)
	kinds = ["A", "B", "A", "B"]
	order = [2, 3, 0, 1]

	assert match_chains(predict(reference, order, rng), reference, kinds) == order


def test_a_bad_prediction_still_pairs_one_to_one():
	rng = np.random.default_rng(2)
	reference = ring([helix(30)], 6, radius=24.0, rng=rng)
	garbage = [rng.normal(scale=30.0, size=chain.shape) for chain in reference]

	assert sorted(match_chains(garbage, reference, ["A"] * 6)) == list(range(6))


def test_chains_that_cannot_pair_as_named_are_left_as_named():
	rng = np.random.default_rng(3)
	reference = ring([helix(30)], 3, radius=12.0, rng=rng)
	prediction = [reference[0], reference[2][:25], reference[1]]

	assert match_chains(prediction, reference, ["A"] * 3) == [0, 1, 2]


def test_superimposed_rmsd_ignores_a_rigid_motion():
	rng = np.random.default_rng(4)
	coordinates = helix(30)
	rotation, translation = rigid_motion(rng)

	assert superimposed_rmsd(coordinates @ rotation + translation, coordinates) == pytest.approx(0.0, abs=1e-9)
