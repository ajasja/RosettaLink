import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

#rosettalink.init('-fast_restyping -mute all')
rosettalink.init("") # Also prints debug, initialisation, more info about movers, their attributes ...


pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQRSTVWY")


# All available attributes
rfdiff_mover = rosettalink.movers.RFDiffusion.RFDiffusion(contig="[7-20]", num_designs="1", rfdiffusion_path="/ceph/hpc/home/olivierif/prosculpt/sif_files/rfdiff.sif", extra_args="diffuser.T=42", work_dir="TESTNIDIROBJ", delete_dir="true")
rfdiff_mover.apply(pose)

print(f"Pose size (All available attributes): {pose.size()} {len(pose)} {pose.total_residue()}, pose sequence: {pose.sequence()}")

# Only required attributes
rfdiff_mover2 = rosettalink.movers.RFDiffusion.RFDiffusion(contig="[7-20]", num_designs="1", rfdiffusion_path="/ceph/hpc/home/olivierif/prosculpt/sif_files/rfdiff.sif", delete_dir="true")
rfdiff_mover2.apply(pose)

print(f"Pose size (Only required attributes): {pose.size()} {len(pose)} {pose.total_residue()}, pose sequence: {pose.sequence()}")