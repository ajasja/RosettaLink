# @file movers/HelloWorldMetric.py
# @brief Basic example of how to implement a SimpleMetric in python.
# @author Moritz Ertelt

import pyrosetta
from rosettalink.decorators import register_metric


class HelloWorldMetric(pyrosetta.rosetta.core.simple_metrics.RealMetric):
    clones_ = list()

    def __init__(self):
        pyrosetta.rosetta.core.simple_metrics.RealMetric.__init__(self)
        self.number_ = None

    def clone(self):
        copy = HelloWorldMetric()
        copy.number_ = self.number_
        HelloWorldMetric.clones_.append(copy)
        return copy

    def name(self):
        return self.class_name()

    def metric(self):
        return self.name()

    def calculate(self, pose):
        return self.number_

    def parse_my_tag(self, tag, datamap):
        self.number_ = tag.get_option_real("number")

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xsct_real

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        attrlist.append(XMLSchemaAttribute.required_attribute(
            "number",
            XMLSchemaType(xsct_real),
            "Number to return when calculate is being called"))

        description = '''
        HelloWorldMetric used to demonstrate 
'''

        pyrosetta.rosetta.core.simple_metrics.xsd_simple_metric_type_definition_w_attributes(
            xsd,
            cls.class_name(),
            description, attrlist)

    def get_name(self):
        return self.class_name()

    @staticmethod
    def class_name():
        return "HelloWorldMetric"


@register_metric
class HelloWorldMetricCreator(pyrosetta.rosetta.core.simple_metrics.SimpleMetricCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.core.simple_metrics.SimpleMetricCreator.__init__(self)

    def create_simple_metric(self):
        instance = HelloWorldMetric()
        self.instances_.append(instance)
        return instance

    def keyname(self):
        return HelloWorldMetric.class_name()

    def provide_xml_schema(self, xsd):
        HelloWorldMetric.provide_xml_schema(xsd)


