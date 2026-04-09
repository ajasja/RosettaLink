# @file pyrosetta_scripts/decorators.py
# @brief Decorators used to register external RosettaScripts components.
# @author Moritz Ertelt

from .PyRosettaScripts import pyrosetta_scripts


def register_mover(cls):
    pyrosetta_scripts.register_movers([cls])
    return cls


def register_metric(cls):
    pyrosetta_scripts.register_metrics([cls])
    return cls


def register_taskop(cls):
    pyrosetta_scripts.register_taskops([cls])
    return cls


def register_residue_selector(cls):
    pyrosetta_scripts.register_residue_selectors([cls])
    return cls
