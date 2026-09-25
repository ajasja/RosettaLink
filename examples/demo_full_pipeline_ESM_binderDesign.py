# @file examples/demo_full_pipeline_ESM_binderDesign.py
# @brief Binder design pipeline: RFDiffusion builds a de novo binder against a
#        fixed target chain, LigandMPNN designs sequences for it, ESMFold2
#        refolds the complex, and five self-consistency RMSDs are reported
#        per design in one CSV.
#
# Same pipeline as demo_full_pipeline_boltz_binderDesign.py with ESMFold2 in
# place of Boltz2. The two are interchangeable at the mover tag, but their
# score keys are not: ESMFold2 writes ESMFold2_* and Boltz2 writes Boltz_*,
# so METRIC_KEYS below differs, and the confidence columns of a CSV from one
# are not directly comparable to those from the other.
#
# The contig "[B1-150/0 90-120]" keeps target chain B (residues 1-150) whole,
# then "/0" starts a new chain of 90-120 de novo residues - the binder.
# Because chain B is kept whole, RFDiffusion writes receptor_con_hal_idx0 to
# the .trb and the RFDiffusion mover labels those residues fixed_chain, the
# de novo binder residues designed, and any kept-backbone residues outside a
# fixed chain motif.
#
# The RMSDs compare the ESMFold2 prediction against the pose ESMFold2 was
# handed (the RFDiffusion backbone carrying the LigandMPNN sequence). Each is
# superimposed and measured over the labels shown:
#                            superimposed on   measured on
#   scRMSD_all               whole complex     whole complex
#   scRMSD_fixed_chains      target            target
#   scRMSD_motif             motif             motif
#   scRMSD_designed          binder            binder
#   scRMSD_binder_on_target  target            binder
#
# All five are alpha-carbon RMSDs (the <RMSD> default, atoms="ca"); set
# atoms="heavy" or atoms="all" on a tag to include sidechains.
#
# scRMSD_designed reports whether the binder folds as designed; only
# scRMSD_binder_on_target also reports whether it is placed against the
# target as designed. It is always the larger of the two, since aligning on
# the binder is by definition the best alignment for measuring the binder.
#
# scRMSD_motif is expected to be absent for this particular contig: the
# target is a whole fixed chain and the binder is fully de novo, so no
# residue is labelled motif. The mover skips an RMSD whose label matches
# nothing and warns, so the column is simply blank. It becomes meaningful
# with a motif-scaffolding contig.

import csv
import os
from pathlib import Path

import pyrosetta
import rosettalink
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from rosettalink.utils import drain_additional_output

# --------------------------------------------------------------------------- #
# Configuration - set the tool paths and command prefixes for your
# installation here; they are substituted into the mover tags in
# PIPELINE_XML_TEMPLATE below.
# --------------------------------------------------------------------------- #
RFDIFFUSION_SIF = "/home/folivieri/prosculpt/singularity_files/rfdiff.sif"
LIGANDMPNN_SIF = "/home/folivieri/prosculpt/singularity_files/ligandmpnn.sif"
# biohub/ESMFold2-Fast is single sequence and quicker; biohub/ESMFold2 is the
# full model. Run this script from an environment with esm and PyRosetta.
ESMFOLD2_MODEL = "biohub/ESMFold2-Fast"

TARGET_PDB = Path("examples/input_data/insulin_target.pdb").resolve()

NUM_BINDERS = 2                # RFDiffusion num_designs
NUM_SEQUENCES_PER_BINDER = 2   # LigandMPNN batch_size (per backbone)
CONTIG = "[B1-150/0 90-120]"   # target chain B kept whole, then a de novo binder

OUTPUT_DIR = Path(os.environ.get("PIPELINE_OUTPUT_DIR", "pipeline_binder_output_esm_3")).resolve()
METRICS_CSV = OUTPUT_DIR / "metrics.csv"

# --------------------------------------------------------------------------- #
# One protocol, top to bottom:
#   make_binder             (RFDiffusion, N binder backbones on the target)
#   -> design_sequences      (MultiplePoseMover: collects all N complexes)
#        -> make_sequence    (LigandMPNN, M sequences per complex)
#        -> predict_structures (MultiplePoseMover: collects all M sequences)
#             -> predict_structure (ESMFold2; refolds the complex, 5 scRMSDs)
#
# num_diffusion_samples stays at 1: raising it fans out again, and those
# extra samples would need one more MultiplePoseMover around
# predict_structure to reach the CSV.
# --------------------------------------------------------------------------- #
PIPELINE_XML_TEMPLATE = """
<ROSETTASCRIPTS>

    <MOVERS>
        <RFDiffusion name="make_binder"
            contig="{contig}"
            num_designs="{num_binders}"
            rfdiffusion_path="{rfdiffusion_path}"
            extra_args="inference.input_pdb=/output/input.pdb"
            work_dir="{rfdiffusion_work_dir}"
            delete_dir="false" />

        <MultiplePoseMover name="design_sequences">
            <ROSETTASCRIPTS>
                <MOVERS>
                    <LigandMPNN name="make_sequence"
                        ligandmpnn_path="{ligandmpnn_path}"
                        batch_size="{num_sequences_per_binder}"
                        work_dir="{ligandmpnn_work_dir}"
                        delete_dir="false" />

                    <MultiplePoseMover name="predict_structures">
                        <ROSETTASCRIPTS>
                            <MOVERS>
                                <ESMFold2 name="predict_structure"
                                    model="{model}"
                                    device="cuda"
                                    num_loops="5"
                                    num_sampling_steps="50"
                                    num_diffusion_samples="1"
                                    prefix_name="ESMFold2_"
                                    replace_pose="1"
                                    work_dir="{esmfold2_work_dir}"
                                    delete_dir="false">
                                    <RMSD name="scRMSD_all" reslabel_input="all" reslabel_prediction="all" />
                                    <RMSD name="scRMSD_fixed_chains" reslabel_input="fixed_chain" reslabel_prediction="fixed_chain" />
                                    <RMSD name="scRMSD_motif" reslabel_input="motif" reslabel_prediction="motif" />
                                    <RMSD name="scRMSD_designed" reslabel_input="designed" reslabel_prediction="designed" />
                                    <RMSD name="scRMSD_binder_on_target" reslabel_input="designed" reslabel_prediction="designed" reslabel_superimpose="fixed_chain" />
                                </ESMFold2>
                            </MOVERS>
                            <PROTOCOLS>
                                <Add mover="predict_structure" />
                            </PROTOCOLS>
                        </ROSETTASCRIPTS>
                    </MultiplePoseMover>
                </MOVERS>
                <PROTOCOLS>
                    <Add mover="make_sequence" />
                    <Add mover="predict_structures" />
                </PROTOCOLS>
            </ROSETTASCRIPTS>
        </MultiplePoseMover>
    </MOVERS>

    <PROTOCOLS>
        <Add mover="make_binder" />
        <Add mover="design_sequences" />
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

# CSV column -> cached score key. Every confidence metric ESMFold2 reports is
# cached as ESMFold2_<name> (plddt, ptm, iptm, and pae or pde when the model
# produces them), so any of those can be added here. pLDDT is reported on
# whatever scale ESMFold2 produces, not rescaled to match AF2 or Boltz.
METRIC_KEYS = {
    "plddt": "ESMFold2_plddt",
    "ptm": "ESMFold2_ptm",
    "iptm": "ESMFold2_iptm",
    "scRMSD_all": "ESMFold2_scRMSD_all",
    "scRMSD_fixed_chains": "ESMFold2_scRMSD_fixed_chains",
    "scRMSD_motif": "ESMFold2_scRMSD_motif",
    "scRMSD_designed": "ESMFold2_scRMSD_designed",
    "scRMSD_binder_on_target": "ESMFold2_scRMSD_binder_on_target",
}

# --------------------------------------------------------------------------- #

rosettalink.init(set_logging_handler="interactive")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not TARGET_PDB.is_file():
    raise SystemExit(f"Target structure not found: {TARGET_PDB}")

xml_string = PIPELINE_XML_TEMPLATE.format(
    rfdiffusion_path=RFDIFFUSION_SIF,
    ligandmpnn_path=LIGANDMPNN_SIF,
    model=ESMFOLD2_MODEL,
    contig=CONTIG,
    num_binders=NUM_BINDERS,
    rfdiffusion_work_dir=OUTPUT_DIR / "rfdiffusion",
    num_sequences_per_binder=NUM_SEQUENCES_PER_BINDER,
    ligandmpnn_work_dir=OUTPUT_DIR / "ligandmpnn",
    esmfold2_work_dir=OUTPUT_DIR / "esmfold2",
)

print(f"\n=== Designing {NUM_BINDERS} binder(s) x {NUM_SEQUENCES_PER_BINDER} sequence(s) against {TARGET_PDB.name} ===")

xml = XmlObjects.create_from_string(xml_string)
protocol = xml.get_mover("ParsedProtocol")

# Unlike the unconditional pipelines, the starting pose matters here: the
# RFDiffusion mover dumps it to input.pdb, which the contig refers to by
# chain (B1-150) via inference.input_pdb in extra_args.
target_pose = pyrosetta.pose_from_file(str(TARGET_PDB))
print(f"Target loaded: {target_pose.total_residue()} residues, {target_pose.num_chains()} chain(s)")

protocol.apply(target_pose)

all_poses = [target_pose] + drain_additional_output(protocol)
print(f"Produced {len(all_poses)} final complex(es) (expected {NUM_BINDERS * NUM_SEQUENCES_PER_BINDER})")

print(f"\n=== Writing structures and metrics to {OUTPUT_DIR} ===")

rows = []
for i, final_pose in enumerate(all_poses):
    predicted_pdb = OUTPUT_DIR / f"binder_{i}.pdb"
    final_pose.dump_pdb(str(predicted_pdb))

    cached = dict(final_pose.cache.fast_items())

    missing = [key for key in METRIC_KEYS.values() if key not in cached]
    if missing:
        print(
            f"WARNING: design {i} is missing expected score(s) {missing}.\n"
            f"\tAvailable keys: {sorted(cached)}"
        )

    row = {"design_index": i, "sequence": final_pose.sequence()}
    row.update({column: cached.get(key) for column, key in METRIC_KEYS.items()})
    row["pdb"] = str(predicted_pdb)
    rows.append(row)
    print(
        f"Design {i}: pLDDT={row['plddt']}, ptm={row['ptm']}, iptm={row['iptm']}, "
        f"scRMSD_all={row['scRMSD_all']}, scRMSD_fixed_chains={row['scRMSD_fixed_chains']}, "
        f"scRMSD_motif={row['scRMSD_motif']}, scRMSD_designed={row['scRMSD_designed']}, "
        f"scRMSD_binder_on_target={row['scRMSD_binder_on_target']}\n"
        f"\twritten to: {predicted_pdb}"
    )

with open(METRICS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n=== Metrics CSV written to {METRICS_CSV} ===")
