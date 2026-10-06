# @file movers/BaseLinkMover.py
# @brief Rosetta mover to prepare temp and work directories.

import pyrosetta


class BaseLinkMover(pyrosetta.rosetta.protocols.moves.Mover):
    def apply(self, pose):
        self.tracer_debug << f"Hello there! \n" and self.tracer_debug.flush()
        
