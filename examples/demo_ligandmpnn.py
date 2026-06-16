import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

#rosettalink.init('-fast_restyping -mute all')
#rosettalink.init('') # Also prints debug, initialisation, more info about movers, their attributes ...
rosettalink.init(set_logging_handler='interactive')


pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQRSTVWY")

# All available attributes
xml_string = """
<ROSETTASCRIPTS>

    <SIMPLE_METRICS>
    </SIMPLE_METRICS>    

    <RESIDUE_SELECTORS>
        <StoredResidueSubset name="get_not_de_novo_residues" subset_name="inpaint_seq" />
        <Not name="get_de_novo_residues" selector="get_not_de_novo_residues" />
    </RESIDUE_SELECTORS>

    <MOVERS>
        <RFDiffusion name="make_backbone" contig="[3-4/0 5-6/A1-10/3/A15-16/2-2/0 A17-20]" num_designs="3" rfdiffusion_path="/home/folivieri/prosculpt/singularity_files/rfdiff.sif" extra_args="inference.input_pdb=/output/input.pdb contigmap.inpaint_seq=[A5-18]" delete_dir="true" work_dir="TESTNIDIRveč" />
        <LigandMPNN name="make_sequence" ligandmpnn_path="/home/zznidar/ligandmpnn/ligandmpnn.sif" delete_dir="true" work_dir="TESTNIDIRveč" />
    </MOVERS>       

    <PROTOCOLS>
        <Add mover="make_backbone"/>
        <Add mover="make_sequence"/>
    </PROTOCOLS>

</ROSETTASCRIPTS>
""" 

# Parse the XML
xml = XmlObjects.create_from_string(xml_string)
protocol = xml.get_mover("ParsedProtocol")
protocol.apply(pose)
print(f"Pose size (All available attributes): {pose.size()} {len(pose)} {pose.total_residue()}, pose sequence: {pose.sequence()}")
pose.dump_pdb("pose_after_PoseFromSequence2.pdb")

# pdb_info() is cleared during PoseFromSequenceMover, so we may want to set the lagels again. 
