# @file examples/demo_full_pipeline_single_xml_extended.py
# @brief Same single-XML pipeline as demo_full_pipeline_single_xml.py
#        (RFDiffusion -> LigandMPNN -> BOLTZ, fully fanned out through
#        nested MultiplePoseMovers), extended to compute six metrics per
#        final design and report them all in one CSV:
#          A) self-consistency RMSD (scRMSD): RMSD of the BOLTZ-predicted
#             structure back to the original RFDiffusion backbone it is
#             based on. LigandMPNN only changes residue identity, never
#             backbone coordinates, so the pose BOLTZ is handed (its own
#             "input_pose", captured right before folding) IS that original
#             backbone - comparing against it is exactly self-consistency.
#          B) pLDDT of the BOLTZ model.
#          C) longest continuous apolar segment (surface hydrophobic patch length).
#          D) net charge.
#          E) buried unsatisfied hydrogen bonds.
#          F) SAP score (spatial aggregation propensity).
#
# A) and B) are computed by the BOLTZ mover itself (it already has a
# built-in <RMSD> sub-tag and always reports pLDDT) - the only piece that was
# actually missing was a residue label spanning the WHOLE chain to RMSD over:
# RFDiffusion's inpaint_seq/inpaint_str labels are both empty for a fully
# unconditional design (nothing is "kept" from an input), so RFDiffusion.py
# now also stamps every residue with an "all" label (same convention already
# used by hand in examples/demo_BOLTZ.py) that survives through
# LigandMPNN into BOLTZ's RMSD comparison.
#
# C), D), E) are native Rosetta filters (LongestContinuousApolarSegment,
# NetCharge, BuriedUnsatHbonds2) added to the innermost sub-protocol with
# confidence="0" - confidence=0 means "never reject the trajectory on this
# filter", while ParsedProtocol still caches its computed value into the
# pose's extra scores under the filter's own name, the same way any Rosetta
# score-file column is produced from a confidence=0 filter in a normal
# design pipeline. Their exact schemas were supplied directly (not guessed).
#
# F) SapScoreMetric is a SimpleMetric, not a Filter, so it needs a
# RunSimpleMetrics mover to actually cache its value onto the pose the same
# way the confidence=0 filters do for C/D/E.
#
# CONFIRMED WORKING (a real run's log): RunSimpleMetrics + SapScoreMetric,
# LongestContinuousApolarSegment, and NetCharge all ran, reported, and their
# values were cached onto the pose exactly as expected -
# "Set filter value for netcharge to: -3" in the log is ParsedProtocol doing
# that caching for a confidence=0 filter, confirmed in practice, not just
# assumed.
#
# BuriedUnsatHbonds2 needs jump_number="0" explicitly on a single-chain pose.
# Its default (1) is meant for interface analysis across a specific jump in
# the FoldTree; a monomer built via pose_from_sequence has zero jumps, and
# the filter asserts jump_num_ <= pose.num_jump() - so the default crashed
# with "Assertion `jump_num_ <= pose.num_jump()` failed." the first time this
# actually ran (protocols/buns/BuriedUnsatHbondFilter2.cc:142), exactly as
# flagged as a risk before that run. jump_number="0" fixes it.

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
RFDIFFUSION_SIF = "/path/to/rfdiff.sif"
LIGANDMPNN_SIF = "/path/to/ligandmpnn.sif"
# boltz is a plain CLI: "boltz predict" if on PATH, or
# "conda run -n <env> boltz predict" to run it from another environment.
BOLTZ_CMD_HEADER = "boltz predict"
BOLTZ_CACHE = "/home/folivieri/.boltz"
NUM_BACKBONES = 2              # RFDiffusion num_designs
NUM_SEQUENCES_PER_BACKBONE = 3 # LigandMPNN batch_size (per backbone)
BACKBONE_LENGTH = 60           # residues; unconditional design (no motif)
CONTIG = f"[{BACKBONE_LENGTH}-{BACKBONE_LENGTH}]"

OUTPUT_DIR = Path(os.environ.get("PIPELINE_OUTPUT_DIR", "pipeline_boltz_output")).resolve()
METRICS_CSV = OUTPUT_DIR / "metrics.csv"

# --------------------------------------------------------------------------- #
# One protocol, top to bottom:
#   make_backbone            (RFDiffusion, N backbones)
#   -> design_sequences       (MultiplePoseMover: collects all N backbones)
#        -> make_sequence     (LigandMPNN, M sequences per backbone)
#        -> predict_structures (MultiplePoseMover: collects all M sequences)
#             -> predict_structure (BOLTZ; reports pLDDT + scRMSD)
#             -> run_sap           (RunSimpleMetrics; reports sap_score)
#             -> apolar_segment, netcharge, buried_unsat (confidence=0 filters; report only)
# --------------------------------------------------------------------------- #
PIPELINE_XML_TEMPLATE = """
<ROSETTASCRIPTS>

    <MOVERS>
        <RFDiffusion name="make_backbone"
            contig="{contig}"
            num_designs="{num_backbones}"
            rfdiffusion_path="{rfdiffusion_path}"
            work_dir="{rfdiffusion_work_dir}"
            delete_dir="false" />

        <MultiplePoseMover name="design_sequences">
            <ROSETTASCRIPTS>
                <MOVERS>
                    <LigandMPNN name="make_sequence"
                        ligandmpnn_path="{ligandmpnn_path}"
                        batch_size="{num_sequences_per_backbone}"
                        work_dir="{ligandmpnn_work_dir}"
                        delete_dir="false" />

                    <MultiplePoseMover name="predict_structures">
                        <ROSETTASCRIPTS>
                            <SCOREFXNS>
                                <ScoreFunction name="sfxn" weights="ref2015" />
                            </SCOREFXNS>
                            <RESIDUE_SELECTORS>
                                <True name="full_pose" />
                            </RESIDUE_SELECTORS>
                            <SIMPLE_METRICS>
                                <SapScoreMetric name="sap_score"
                                    score_selector="full_pose"
                                    sap_calculate_selector="full_pose"
                                    sasa_selector="full_pose" />
                            </SIMPLE_METRICS>
                            <MOVERS>
                                <Boltz2 name="predict_structure"
                                    cmd_header="{cmd_header}"
                                    cache="{cache}"
                                    use_msa_server="false"
                                    diffusion_samples="1"
                                    prefix_name="Boltz_"
                                    replace_pose="1"
                                    work_dir="{BOLTZ_work_dir}"
                                    delete_dir="false">
                                    <RMSD name="scRMSD" reslabel_input="all" reslabel_prediction="all" />
                                </Boltz2>
                                <RunSimpleMetrics name="run_sap" metrics="sap_score" />
                            </MOVERS>
                            <FILTERS>
                                <LongestContinuousApolarSegment name="apolar_segment"
                                    residue_selector="full_pose"
                                    confidence="0" />
                                <NetCharge name="netcharge" confidence="0" />
                                <BuriedUnsatHbonds2 name="buried_unsat"
                                    scorefxn="sfxn"
                                    jump_number="0"
                                    confidence="0" />
                            </FILTERS>
                            <PROTOCOLS>
                                <Add mover="predict_structure" />
                                <Add mover="run_sap" />
                                <Add filter_name="apolar_segment" />
                                <Add filter_name="netcharge" />
                                <Add filter_name="buried_unsat" />
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
        <Add mover="make_backbone" />
        <Add mover="design_sequences" />
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

# --------------------------------------------------------------------------- #

rosettalink.init(set_logging_handler="interactive")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

xml_string = PIPELINE_XML_TEMPLATE.format(
    rfdiffusion_path=RFDIFFUSION_SIF,
    ligandmpnn_path=LIGANDMPNN_SIF,
    cmd_header=BOLTZ_CMD_HEADER,
    cache=BOLTZ_CACHE,
    contig=CONTIG,
    num_backbones=NUM_BACKBONES,
    rfdiffusion_work_dir=OUTPUT_DIR / "rfdiffusion",
    num_sequences_per_backbone=NUM_SEQUENCES_PER_BACKBONE,
    ligandmpnn_work_dir=OUTPUT_DIR / "ligandmpnn",
    BOLTZ_work_dir=OUTPUT_DIR / "BOLTZ",
)

print(f"\n=== Running single-XML pipeline: {NUM_BACKBONES} backbone(s) x {NUM_SEQUENCES_PER_BACKBONE} sequence(s) each ===")

xml = XmlObjects.create_from_string(xml_string)
protocol = xml.get_mover("ParsedProtocol")

# Starting pose is only used to write a placeholder input.pdb; since no
# inference.input_pdb/contigmap motif is given, RFDiffusion ignores its
# contents entirely and diffuses de novo backbones from noise.
seed_pose = pyrosetta.pose_from_sequence("A" * BACKBONE_LENGTH)
protocol.apply(seed_pose)

all_poses = [seed_pose] + drain_additional_output(protocol)
print(f"Produced {len(all_poses)} final structure(s) (expected {NUM_BACKBONES * NUM_SEQUENCES_PER_BACKBONE})")

print(f"\n=== Writing structures and metrics to {OUTPUT_DIR} ===")

# CSV column -> cached score key. Boltz2 prefixes its scores with
# prefix_name ("Boltz_") and reports pLDDT as complex_plddt on a 0-1 scale;
# the filters and SimpleMetric are keyed by their own XML names.
METRIC_KEYS = {
    "plddt": "Boltz_complex_plddt",
    "scRMSD": "Boltz_scRMSD",
    "apolar_segment_length": "apolar_segment",
    "net_charge": "netcharge",
    "buried_unsat_hbonds": "buried_unsat",
    "sap_score": "sap_score",
}

rows = []
for i, final_pose in enumerate(all_poses):
    predicted_pdb = OUTPUT_DIR / f"result_{i}.pdb"
    final_pose.dump_pdb(str(predicted_pdb))

    # Pose.cache replaces the deprecated Pose.scores. Flattening it once per
    # pose covers all three namespaces (extra scores from the movers and the
    # confidence=0 filters, SimpleMetric data, energies) in one lookup table.
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
        f"Result {i}: pLDDT={row['plddt']}, scRMSD={row['scRMSD']}, "
        f"apolar_segment_length={row['apolar_segment_length']}, net_charge={row['net_charge']}, "
        f"buried_unsat_hbonds={row['buried_unsat_hbonds']}, sap_score={row['sap_score']}\n"
        f"\tsequence={row['sequence']}\n"
        f"\twritten to: {predicted_pdb}"
    )

with open(METRICS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n=== Metrics CSV written to {METRICS_CSV} ===")
