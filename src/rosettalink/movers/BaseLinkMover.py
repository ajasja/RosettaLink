# @file movers/BaseLinkMover.py
# @brief Rosetta mover to prepare temp and work directories.

import os
import shutil

import pyrosetta
from rosettalink.decorators import register_mover
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer


import tempfile
from pathlib import Path


class BaseLinkMover(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def apply(self, pose):
        self.tracer_debug << f"Hello there! \n" and self.tracer_debug.flush()
        


    def get_name(self):
        return self.mover_name()

   


    @staticmethod
    def mover_name():
        return "BaseLinkMover"

    @classmethod
    def provide_xml_schema(cls, xsd):
        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        description = '''
                        Your mover's parent to automatically create a temp dir, cd to there (so singularity containers automatically bind needed folders), and then cd back to the original working directory.
                      '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd,
            cls.mover_name(),
            description, attrlist)


@register_mover
class BaseLinkMoverCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = BaseLinkMover()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return BaseLinkMover.mover_name()

    def provide_xml_schema(self, xsd):
        BaseLinkMover.provide_xml_schema(xsd)

