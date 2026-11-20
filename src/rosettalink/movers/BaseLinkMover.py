# @file movers/BaseLinkMover.py
# @brief Base class of every mover that wraps an external program.

import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.select.residue_selector import get_residue_selector
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

    @staticmethod
    def selector_option(tag, option, data):
        """The residue selector an attribute names, or None when unset, which
        every caller reads as every residue."""
        if not tag.hasOption(option):
            return None
        return get_residue_selector(tag.get_option_string(option), data)

    @staticmethod
    def selected_positions(pose, selector):
        """Residues a selector picks, in pose numbering. A selector of None is
        every residue."""
        if selector is None:
            return list(range(1, pose.total_residue() + 1))
        picked = selector.apply(pose)
        return [resnum for resnum in range(1, pose.total_residue() + 1) if picked[resnum]]

    @staticmethod
    def index_selector(resnums):
        return ResidueIndexSelector(",".join(str(resnum) for resnum in resnums))

    def parse_rmsd_tag(self, child, data):
        """One RMSD subtag.

        calculation_selector names the residues the RMSD is measured over and
        alignment_selector the residues the structures are superimposed on.
        Both name a residue selector and both default to every residue, so an
        RMSD subtag with neither measures the whole prediction."""
        return {
            "name": child.get_option_string("name"),
            "calculation": self.selector_option(child, "calculation_selector", data),
            "alignment": self.selector_option(child, "alignment_selector", data),
            "atoms": child.get_option_string("atoms") if child.hasOption("atoms") else "ca",
        }

    def parse_metric_tag(self, child, data, quantities):
        """One metric subtag: base_metric names the per residue confidence
        value, one of quantities, and selector the residues it is averaged
        over, defaulting to every residue."""
        base_metric = child.get_option_string("base_metric").lower()
        if base_metric not in quantities:
            raise RuntimeError(
                f"metric {child.get_option_string('name')} has base_metric="
                f"{child.get_option_string('base_metric')}, which is not one of "
                f"{', '.join(quantities)}"
            )
        return {
            "name": child.get_option_string("name"),
            "quantity": base_metric,
            "selector": self.selector_option(child, "selector", data),
        }

    def averaged_over_selection(self, pose, selector, values):
        """One confidence quantity averaged over the residues a selector
        picks. A residue by residue matrix is averaged over the block where
        both residues are selected. None when the selection is empty."""
        resnums = [
            resnum
            for resnum in self.selected_positions(pose, selector)
            if resnum <= len(values)
        ]
        if not resnums:
            return None
        first = values[0]
        if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
            picked = [values[i - 1][j - 1] for i in resnums for j in resnums]
        else:
            picked = [values[resnum - 1] for resnum in resnums]
        return float(sum(float(value) for value in picked) / len(picked))
