# @file examples/demo_full_pipeline.py
# @brief End-to-end demo: RFDiffusion (unconditional backbone generation)
#        -> LigandMPNN (sequence design) -> ColabFold (structure prediction)
#
# Generates NUM_BACKBONES de novo backbones from scratch (no input structure,
# no motif/contig scaffolding), designs a sequence for each with LigandMPNN,
# then predicts the structure of each designed sequence with ColabFold.
#
# NOTE on current mover behaviour: RFDiffusion.apply() only ever pulls the
# *first* (alphabetically sorted) output .pdb file into the pose, even if
# num_designs > 1 (see the "TODO: multi-pose" comment in RFDiffusion.py).
# So to get NUM_BACKBONES independent backbones we call the mover once per
# backbone with num_designs="1", rather than once with num_designs="2".

import os
from pathlib import Path

import pyrosetta
import rosettalink
from rosettalink.movers.RFDiffusion import RFDiffusion
from rosettalink.movers.LigandMPNN import LigandMPNN
from rosettalink.movers.ColabFold import ColabFold

# --------------------------------------------------------------------------- #
# Configuration - adjust these paths for your site/installation.
# Each can also be overridden with an environment variable of the same name,
# e.g. RFDIFFUSION_SIF=/path/to/rfdiff.sif python demo_full_pipeline.py
# --------------------------------------------------------------------------- #
RFDIFFUSION_SIF = os.environ.get(
    "RFDIFFUSION_SIF", "/home/folivieri/prosculpt/singularity_files/rfdiff.sif"
)
LIGANDMPNN_SIF = os.environ.get(
    "LIGANDMPNN_SIF", "/home/folivieri/prosculpt/singularity_files/ligandmpnn.sif"
)
COLABFOLD_CMD_HEADER = os.environ.get(
    "COLABFOLD_CMD_HEADER",
    f"singularity run --nv "
    f"{os.environ.get('COLABFOLD_SIF', '/home/folivieri/prosculpt/singularity_files/colabfold.sif')} "
    f"colabfold_batch",
)

NUM_BACKBONES = 2
BACKBONE_LENGTH = 100          # residues; unconditional design (no motif)
CONTIG = f"[{BACKBONE_LENGTH}-{BACKBONE_LENGTH}]"

OUTPUT_DIR = Path(os.environ.get("PIPELINE_OUTPUT_DIR", "pipeline_output")).resolve()

# --------------------------------------------------------------------------- #

rosettalink.init(set_logging_handler="interactive")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []

for i in range(NUM_BACKBONES):
    design_dir = OUTPUT_DIR / f"design_{i}"
    design_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Design {i}: generating backbone ({BACKBONE_LENGTH} residues, unconditional) ===")

    # Starting pose is only used to write a placeholder input.pdb; since no
    # inference.input_pdb/contigmap motif is given, RFDiffusion ignores its
    # contents entirely and diffuses a de novo backbone from noise.
    pose = pyrosetta.pose_from_sequence("A" * BACKBONE_LENGTH)

    rfdiff_mover = RFDiffusion(
        contig=CONTIG,
        num_designs="1",
        rfdiffusion_path=RFDIFFUSION_SIF,
        work_dir=str(design_dir / "rfdiffusion"),
        delete_dir=False,
    )
    rfdiff_mover.apply(pose)
    backbone_pdb = design_dir / "backbone.pdb"
    pose.dump_pdb(str(backbone_pdb))
    print(f"Design {i}: backbone written to {backbone_pdb} ({pose.total_residue()} residues)")

    print(f"=== Design {i}: designing sequence with LigandMPNN ===")
    ligandmpnn_mover = LigandMPNN(
        ligandmpnn_path=LIGANDMPNN_SIF,
        work_dir=str(design_dir / "ligandmpnn"),
        delete_dir=False,
    )
    ligandmpnn_mover.apply(pose)
    designed_pdb = design_dir / "designed.pdb"
    pose.dump_pdb(str(designed_pdb))
    designed_sequence = pose.sequence()
    print(f"Design {i}: designed sequence written to {designed_pdb}")
    print(f"Design {i}: sequence = {designed_sequence}")

    print(f"=== Design {i}: predicting structure with ColabFold ===")
    colabfold_mover = ColabFold(
        cmd_header=COLABFOLD_CMD_HEADER,
        models="1,2,3,4,5",
        msa_mode="single_sequence",
        rank="auto",
        prefix_name=f"AF2_{i}_",
        replace_pose=True,
        work_dir=str(design_dir / "colabfold"),
        delete_dir=False,
    )
    colabfold_mover.apply(pose)
    predicted_pdb = design_dir / "predicted.pdb"
    pose.dump_pdb(str(predicted_pdb))

    plddt = pose.scores.get(f"AF2_{i}_plddt")
    pae = pose.scores.get(f"AF2_{i}_pae")
    print(f"Design {i}: prediction written to {predicted_pdb}")
    print(f"Design {i}: mean pLDDT = {plddt}, mean PAE = {pae}")

    results.append({
        "index": i,
        "sequence": designed_sequence,
        "backbone_pdb": str(backbone_pdb),
        "designed_pdb": str(designed_pdb),
        "predicted_pdb": str(predicted_pdb),
        "plddt": plddt,
        "pae": pae,
    })

print("\n=== Pipeline summary ===")
for r in results:
    print(
        f"Design {r['index']}: pLDDT={r['plddt']}, PAE={r['pae']}, "
        f"sequence={r['sequence']}\n\tpredicted structure: {r['predicted_pdb']}"
    )
