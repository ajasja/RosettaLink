import os

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
        <RFdiffusion name="make_backbone" contig="[3-4/0 5-6/A1-10/3/A15-16/2-2/0 A17-20]" num_designs="1" rfdiffusion_path="~/shared_folder/sif_files/rfdiff.sif" extra_args="inference.input_pdb=input.pdb contigmap.inpaint_seq=[A5-18]" delete_dir="true" work_dir="/full/absolute/path/TESTNIDIRIRinpaintMASKED" />
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
print("Dumping pose to file: TESTNIDIRIRinpaintMASKED/TESTNIDIRIRinpaintMASKED_00013.pdb")
os.makedirs("TESTNIDIRIRinpaintMASKED", exist_ok=True)
pose.dump_pdb("TESTNIDIRIRinpaintMASKED/TESTNIDIRIRinpaintMASKED_00013.pdb")

# Residues to mutate are also saved in pose.pdb_info().res_haslabel(1, "inpaint_seq")
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
mutate_residue = MutateResidue()
mutate_residue.set_res_name("TYR")
for i in range(1, pose.size()+1):
    if pose.pdb_info().res_haslabel(i, "inpaint_seq"):
        print(f"Mutating residue {i} which has label inpaint_seq")
        mutate_residue.set_target(i)
        mutate_residue.apply(pose)
print(f"Pose size after mutating inpaint_seq residues to TYR: {pose.size()} {len(pose)} {pose.total_residue()}, pose sequence: {pose.sequence()}")

pose.dump_pdb("TESTNIDIRIRinpaintMASKED/TESTNIDIRIRinpaintMASKED_00013_mutated.pdb")