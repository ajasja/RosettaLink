import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

rosettalink.init('-fast_restyping -mute all')
#rosettalink.init("") # Also prints debug, initialisation, more info about movers, their attributes ...


pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQRSTVWY")


xml_string = """
<ROSETTASCRIPTS>

    <SIMPLE_METRICS>
        <HelloWorldMetric name="hello_metric" number="42.0"/>
    </SIMPLE_METRICS>    

    <MOVERS>
        <HelloWorldMover name="hello_mover" message="Hello, PyRosetta!"/>
        PhiByXDegreesMover name="phi_mover" residue="1" degrees="15"/> # This line does not start with "kotnik v desno (>)", therefore is treated as a comment by Rosetta.
        <RunSimpleMetrics name="run" metrics="hello_metric"/>
        <CmdMover name="cmd_mover" command="hostname" iknowwhatiamdoing="True"/>
    </MOVERS>       

    <PROTOCOLS>
        <Add mover="hello_mover"/>
        Add mover="phi_mover"/> # This line does not start with "kotnik v desno (>)", therefore is treated as a comment by Rosetta.
        <Add mover="run"/>
        <Add mover="cmd_mover"/>
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

# Parse the XML
xml = XmlObjects.create_from_string(xml_string)
protocol = xml.get_mover("ParsedProtocol")
protocol.apply(pose)
metric_value = pose.scores['HelloWorldMetric']
print(f"Metric value: {metric_value}")