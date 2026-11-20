# @file movers/BaseLinkMover.py
# @brief Base class of every mover that wraps an external program.

import pyrosetta
from pyrosetta.rosetta.protocols.moves import AdditionalPosesMover


class BaseLinkMover(AdditionalPosesMover):
    """Inside a ParsedProtocol the mover is connected to the step before it
    and takes every pose that step produced, not only the one passed to
    apply(). Needs a PyRosetta build carrying AdditionalPosesMover."""

    def take_input_poses(self, pose):
        """A clone of pose followed by every pose the previous step has left,
        or [pose] when there is no previous step, which is the case for the
        first step of a protocol and for an apply() called directly.

        Disconnects the previous step afterwards, so a later apply() of the
        same instance does not pull from it again. Call it before changing
        directory: an intervening FOR_EACH_POSE runs its movers and filters
        while its poses are pulled."""
        if not self.previous_mover_set():
            return [pose]
        poses = [pose.clone()] + list(self.get_next_pose_batch(0))
        self.set_previous_mover(None)
        return poses
