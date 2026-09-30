# @file movers/CmdMover.py
# @brief Basic example of how to implement a Mover in python.
# @author Moritz Ertelt

import os

import pyrosetta
from rosettalink.decorators import register_mover


class CmdMover(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.command_ = "echo YouHaventEneteredACommand"
        self.iknowwhatiamdoing_ = False

    def clone(self):
        copy = CmdMover()
        copy.command_ = self.command_
        copy.iknowwhatiamdoing_ = self.iknowwhatiamdoing_
        CmdMover.clones_.append(copy)
        return copy

    def apply(self, pose):
        print(f"[CMDMOVER] self.iknowwhatiamdoing_: {self.iknowwhatiamdoing_}; its type: {type(self.iknowwhatiamdoing_)}")
        if(self.iknowwhatiamdoing_ != True):
            raise Exception("[CMDMOVER] Are you sure you know what you are doing? Running arbitrary commands may damage your files. \n You have to set iknowwhatiamdoing to true if you want to run this mover. This is a safety mechanism to prevent accidental execution of arbitrary commands.")
        pose.set_psi(7, 42.0)
        print(f"[CMDMOVER] Running command: {self.command_}")
        self.run_and_log(self.command_)


    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        print(f"[CMDMOVER] Parsing my tag @ CmdMover. Self: {self}, tag: {tag}, data: {data}")
        self.command_ = tag.get_option_string("command")
        self.iknowwhatiamdoing_ = tag.get_option_bool("iknowwhatiamdoing")
        print(f"[CMDMOVER] Parsed options: command: {self.command_}, iknowwhatiamdoing: {self.iknowwhatiamdoing_}")
    
    def run_and_log(self, command):
        """Runs a command using os.system and also logs the command before running using print"""
        stat = os.system(command)
        wife = os.WIFEXITED(stat)
        exitCode = os.waitstatus_to_exitcode(stat)
        print(f"[CMDMOVER] Command exited with status {stat} and WIFEXITED {wife}. Exit code: {exitCode}")
        if exitCode != 0:
            print("[CMDMOVER] There was an error running the command. We consider it fatal to prevent any file loss. Check the logs and contact the developer.")
            dodatek = ""

            raise Exception(f"[CMDMOVER] Command exited with exit code {exitCode}\n\n{dodatek}")



    @staticmethod
    def mover_name():
        return "CmdMover"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "command",
            XMLSchemaType(xs_string),
            "Command to run when apply is being called"))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "iknowwhatiamdoing",
            XMLSchemaType(xs_string),
            "Safety mechanism to prevent accidental execution of arbitrary commands. Set this to true if you know what you are doing and want to run the command."))

        description = '''
                        Runs an arbitrary command. 
                        Also sets the phi angle of the first residue to 42 for testing purposes.
                      '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd,
            cls.mover_name(),
            description, attrlist)


@register_mover
class CmdMoverCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = CmdMover()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return CmdMover.mover_name()

    def provide_xml_schema(self, xsd):
        CmdMover.provide_xml_schema(xsd)

