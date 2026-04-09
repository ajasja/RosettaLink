# @file movers/HelloWorldMover2.py
# @brief Basic example of how to implement a Mover in python.
# @author Moritz Ertelt

import pyrosetta
from rosettalink.decorators import register_mover


class HelloWorldMover2(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.message_ = None

    def clone(self):
        copy = HelloWorldMover2()
        copy.message_ = self.message_
        HelloWorldMover2.clones_.append(copy)
        return copy

    def apply(self, pose):
        print(self.message_)
        pose.set_psi(7, 42.0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.message_ = tag.get_option_string("message")

    @staticmethod
    def mover_name():
        return "HelloWorldMover2"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "message",
            XMLSchemaType(xs_string),
            "Message to print when apply is being called"))

        description = '''
                        Really good for nothing besides demonstrating how to subclass a mover in PyRosetta. 
                        Also sets the phi angle of the first residue to 42 for testing purposes.
                      '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd,
            cls.mover_name(),
            description, attrlist)


@register_mover
class HelloWorldMover2Creator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = HelloWorldMover2()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return HelloWorldMover2.mover_name()

    def provide_xml_schema(self, xsd):
        HelloWorldMover2.provide_xml_schema(xsd)

