"""The pose buffer handed between batching movers.

Poses are stood in for by an object exposing the two methods the buffer
inspects, so these run without PyRosetta.
"""

import pytest

from rosettalink.utils import PoseBuffer


class FakePose:
	def __init__(self, sequence):
		self.sequence_ = sequence

	def sequence(self):
		return self.sequence_

	def total_residue(self):
		return len(self.sequence_)


class FakeMover:
	pass


@pytest.fixture
def buffer():
	return PoseBuffer()


def test_an_empty_buffer_consumes_to_the_pose_handed_in(buffer):
	pose = FakePose("AAAA")

	assert buffer.consume(pose) == [pose]


def test_published_poses_are_consumed_by_the_next_mover(buffer):
	producer = FakeMover()
	poses = [FakePose("AAAA"), FakePose("CCCC"), FakePose("DDDD")]
	buffer.publish(producer, poses)

	assert buffer.consume(FakePose("AAAA")) == poses


def test_consuming_empties_the_buffer(buffer):
	producer = FakeMover()
	buffer.publish(producer, [FakePose("AAAA"), FakePose("CCCC")])

	buffer.consume(FakePose("AAAA"))

	assert buffer.pop_for(producer) is None


def test_the_producer_serves_every_pose_beyond_the_primary(buffer):
	producer = FakeMover()
	poses = [FakePose("AAAA"), FakePose("CCCC"), FakePose("DDDD")]
	buffer.publish(producer, poses)

	served = []
	while True:
		pose = buffer.pop_for(producer)
		if pose is None:
			break
		served.append(pose)

	# poses[0] is already in the pose apply() was handed.
	assert served == poses[1:]


def test_a_mover_that_does_not_own_the_buffer_is_served_nothing(buffer):
	producer = FakeMover()
	other = FakeMover()
	buffer.publish(producer, [FakePose("AAAA"), FakePose("CCCC")])

	assert buffer.pop_for(other) is None


def test_publishing_transfers_ownership(buffer):
	producer = FakeMover()
	consumer = FakeMover()
	buffer.publish(producer, [FakePose("AAAA"), FakePose("CCCC")])

	buffer.consume(FakePose("AAAA"))
	buffer.publish(consumer, [FakePose("EEEE"), FakePose("FFFF")])

	assert buffer.pop_for(producer) is None
	assert buffer.pop_for(consumer).sequence() == "FFFF"


def test_a_stale_buffer_is_ignored(buffer):
	# Poses left over from an earlier protocol do not match the pose the next
	# mover is handed.
	buffer.publish(FakeMover(), [FakePose("AAAA"), FakePose("CCCC")])
	pose = FakePose("WWWWWW")

	assert buffer.consume(pose) == [pose]


def test_a_stale_buffer_is_emptied(buffer):
	producer = FakeMover()
	buffer.publish(producer, [FakePose("AAAA"), FakePose("CCCC")])

	buffer.consume(FakePose("WWWWWW"))

	assert buffer.pop_for(producer) is None


def test_a_sequence_of_the_same_length_is_still_matched_by_sequence(buffer):
	buffer.publish(FakeMover(), [FakePose("AAAA"), FakePose("CCCC")])
	pose = FakePose("CCCC")

	assert buffer.consume(pose) == [pose]


def test_drain_returns_what_is_left_and_empties_the_buffer(buffer):
	producer = FakeMover()
	poses = [FakePose("AAAA"), FakePose("CCCC"), FakePose("DDDD")]
	buffer.publish(producer, poses)
	buffer.pop_for(producer)

	assert buffer.drain() == [poses[2]]
	assert buffer.drain() == []


def test_clear_empties_the_buffer(buffer):
	producer = FakeMover()
	buffer.publish(producer, [FakePose("AAAA"), FakePose("CCCC")])

	buffer.clear()

	assert buffer.pop_for(producer) is None
	assert buffer.drain() == []
