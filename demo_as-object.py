import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

#rosettalink.init('-fast_restyping -mute all')
rosettalink.init("") # Also prints debug, initialisation, more info about movers, their attributes ...


pose = pyrosetta.pose_from_sequence("ACDEFGHIKLMNPQRSTVWY")



rfdiff_mover = rosettalink.movers.RFDiffusion.RFDiffusion(contig="[7-20]", num_designs="1", rfdiffusion_path="/ceph/hpc/home/olivierif/prosculpt/sif_files/rfdiff.sif", extra_args="", work_dir="", delete_dir="true")
rfdiff_mover.apply(pose)

print(f"Pose size: {pose.size()} {len(pose)} {pose.total_residue()}, pose sequence: {pose.sequence()}")
