# @file examples/demo_esmfold2.py
# @brief Smoke test for the ESMFold2 mover: folds one short peptide, reports
#        confidence scores and a self-consistency RMSD.
#
# Run from an environment holding esm and PyRosetta; the mover imports esm.
#
# To use ESMFold2 in place of Boltz2 in the pipeline demos, swap the mover
# tag for the block below and change the score keys the CSV reads from
# Boltz_* to ESMFold2_*, or the columns come out blank.
#
#   <ESMFold2 name="predict_structure"
#       model="biohub/ESMFold2-Fast"
#       num_diffusion_samples="1"
#       prefix_name="ESMFold2_"
#       replace_pose="1"
#       work_dir="ESMFOLD2_WORK"
#       delete_dir="false">
#       <RMSD name="scRMSD_all" reslabel_input="all" reslabel_prediction="all" />
#   </ESMFold2>

import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

rosettalink.init(set_logging_handler="interactive")

# --------------------------------------------------------------------------- #
# Configuration - set these for your installation.
# --------------------------------------------------------------------------- #
# biohub/ESMFold2-Fast is single sequence and quicker; biohub/ESMFold2 is the
# full model.
ESMFOLD2_MODEL = "biohub/ESMFold2-Fast"


pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQRSTVWY")
for resnum in range(1, pose.total_residue() + 1):
    pose.pdb_info().add_reslabel(resnum, "all")

xml_string = """
<ROSETTASCRIPTS>

    <MOVERS>
        <ESMFold2 name="predict_structure"
            model="{model}"
            device="cuda"
            num_loops="3"
            num_sampling_steps="50"
            num_diffusion_samples="1"
            prefix_name="ESMFold2_"
            replace_pose="1"
            work_dir="TEST_ESMFOLD2"
            delete_dir="false">
            <RMSD name="scRMSD" reslabel_input="all" reslabel_prediction="all" />
        </ESMFold2>
    </MOVERS>

    <PROTOCOLS>
        <Add mover="predict_structure" />
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

xml = XmlObjects.create_from_string(
    xml_string.format(model=ESMFOLD2_MODEL)
)
protocol = xml.get_mover("ParsedProtocol")
protocol.apply(pose)

pose.dump_pdb("esmfold2_test_prediction.pdb")
cached = dict(pose.cache.fast_items())
for key in ("ESMFold2_plddt", "ESMFold2_ptm", "ESMFold2_iptm", "ESMFold2_pae", "ESMFold2_scRMSD"):
    print(f"{key}: {cached.get(key)}")
