import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

rosettalink.init(set_logging_handler="interactive")

# --------------------------------------------------------------------------- #
# Configuration - set these for your installation.
# --------------------------------------------------------------------------- #
BOLTZ_CMD_HEADER = "boltz predict"
BOLTZ_CACHE = "/home/folivieri/.boltz"


pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQRSTVWY")
for resnum in range(1, pose.total_residue() + 1):
    pose.pdb_info().add_reslabel(resnum, "all")

xml_string = """
<ROSETTASCRIPTS>

    <MOVERS>
        <Boltz2 name="predict_structure"
            cmd_header="{cmd_header}"
            cache="{cache}"
            use_msa_server="true"
            diffusion_samples="1"
            prefix_name="Boltz_"
            replace_pose="1"
            work_dir="TEST_BOLTZ2"
            delete_dir="false">
            <RMSD name="scRMSD" reslabel_input="all" reslabel_prediction="all" />
        </Boltz2>
    </MOVERS>

    <PROTOCOLS>
        <Add mover="predict_structure" />
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

xml = XmlObjects.create_from_string(
    xml_string.format(cmd_header=BOLTZ_CMD_HEADER, cache=BOLTZ_CACHE)
)
protocol = xml.get_mover("ParsedProtocol")
protocol.apply(pose)

pose.dump_pdb("boltz2_test_prediction.pdb")
cached = dict(pose.cache.fast_items())
for key in ("Boltz_confidence_score", "Boltz_complex_plddt", "Boltz_ptm", "Boltz_iptm", "Boltz_scRMSD"):
    print(f"{key}: {cached.get(key)}")
