import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

#rosettalink.init('-fast_restyping -mute all')
#rosettalink.init('')  
rosettalink.init(set_logging_handler='interactive')


pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQRSTVWY")
for resnum in range(1, pose.total_residue() + 1):
    pose.pdb_info().add_reslabel(resnum, "all")
for resnum in range(1, 11):
    pose.pdb_info().add_reslabel(resnum, "sculpted")

xml_string = """
<ROSETTASCRIPTS>

    <MOVERS>
        <ColabFold name="predict_structure" models="1,2,3,4,5" prefix_name="AF2_" replace_pose="1" msa_mode="single_sequence" cmd_header="singularity run --nv /home/folivieri/prosculpt/singularity_files/colabfold.sif colabfold_batch" work_dir="/home/d12-studenti/tandatp/test/" delete_dir="0">
            <RMSD name="rmsd_all" reslabel_input="all" reslabel_prediction="all"/>
            <RMSD name="rmsd_fixed_chain" reslabel_input="fixed_chain" reslabel_prediction="all"/>
            <RMSD name="rmsd_sculpted" reslabel_input="sculpted" reslabel_prediction="all"/>
            <RMSD name="rmsd_motif" reslabel_input="motif" reslabel_prediction="all"/>
        </ColabFold>
    </MOVERS>

    <PROTOCOLS>
        <Add mover="predict_structure"/>
    </PROTOCOLS>

</ROSETTASCRIPTS>
""" 

xml = XmlObjects.create_from_string(xml_string)
protocol = xml.get_mover("ParsedProtocol") 
protocol.apply(pose)
pose.dump_pdb("/home/d12-studenti/tandatp/test/best_prediction.pdb")