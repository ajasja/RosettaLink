# @file movers/BaseLinkMover.py
# @brief Base class of every mover that wraps an external program.

import pyrosetta
from pyrosetta.rosetta.protocols.rosetta_scripts import BatchedMultiplePoseMover
from pyrosetta.rosetta.utility import vector1_std_shared_ptr_core_pose_Pose_t as PoseVector


class BaseLinkMover(BatchedMultiplePoseMover):
    """Subclassing BatchedMultiplePoseMover is what makes ParsedProtocol hand
    the mover every pose of the step before it, rather than only the one
    passed to apply(). It needs a PyRosetta build carrying that class."""

    def take_input_poses(self, pose):
        """Every pose ParsedProtocol drained from the previous step, or [pose]
        when it filled nothing in, which is the case for the first step of a
        protocol and for an apply() called directly.

        Empties the list: it lives on the C++ side of the same mover instance
        and otherwise reaches the next apply() still holding this batch."""
        poses = list(self.batched_poses())
        self.set_batched_poses(PoseVector())
        return poses or [pose]
