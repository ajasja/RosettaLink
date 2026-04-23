import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

#rosettalink.init('-fast_restyping -mute all')
rosettalink.init("") # Also prints debug, initialisation, more info about movers, their attributes ...


pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQRSTVWY")

# All available attributes
xml_string = """
<ROSETTASCRIPTS>

    <SIMPLE_METRICS>
    </SIMPLE_METRICS>    

    <MOVERS>
        <RFDiffusion name="make_backbone" contig="[7-20]" num_designs="1" rfdiffusion_path="/ceph/hpc/home/olivierif/prosculpt/sif_files/rfdiff.sif" extra_args="diffuser.T=42" delete_dir="true" work_dir="TESTNIDIRIR" />
    </MOVERS>       

    <PROTOCOLS>
        <Add mover="make_backbone"/>
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

# Parse the XML
xml = XmlObjects.create_from_string(xml_string)
protocol = xml.get_mover("ParsedProtocol")
protocol.apply(pose)
print(f"Pose size (All available attributes): {pose.size()} {len(pose)} {pose.total_residue()}, pose sequence: {pose.sequence()}")

# Only required attributes
xml_string2 = """
<ROSETTASCRIPTS>

    <SIMPLE_METRICS>
    </SIMPLE_METRICS>    

    <MOVERS>
        <RFDiffusion name="make_backbone" contig="[7-20]" num_designs="1" rfdiffusion_path="/ceph/hpc/home/olivierif/prosculpt/sif_files/rfdiff.sif" delete_dir="true" />
    </MOVERS>       

    <PROTOCOLS>
        <Add mover="make_backbone"/>
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

# Parse the XML
xml2 = XmlObjects.create_from_string(xml_string2)
protocol2 = xml2.get_mover("ParsedProtocol")
protocol2.apply(pose)
print(f"Pose size (Only required attributes): {pose.size()} {len(pose)} {pose.total_residue()}, pose sequence: {pose.sequence()}")
