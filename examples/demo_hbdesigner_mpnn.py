# @file examples/demo_hbdesigner_extended.py
# @brief Designs hydrogen bond networks across the 1YRK interface, redesigns
#        the rest of the designed chain around each network with LigandMPNN,
#        then scores every design with InterfaceAnalyzerMover into one CSV.
#
# This follows the postprocessing HBDesigner itself ships
# (examples/postprocessing/run_mpnn.sh): HBDesigner designs the network
# against glycine neighbours, then LigandMPNN redesigns and packs every other
# position on the designed chain with the network held fixed. Grafting the
# native sequence back without that redesign leaves the native rotamers
# clashing with the new network sidechains.
#
# The network is held fixed through fixed_reslabel="hbnet", the label
# HBDesigner stamps, so no residue list has to be written by hand.
# pack_side_chains builds the sidechains that the interface metrics need.
#
# chains_to_design is the designed side only, so the peptide keeps its native
# sequence. Note that everything else on that chain is redesigned, so a
# design differs from the input at many more positions than the network -
# mutations_vs_input in the CSV reports how many.
#
# The same analyzer is run on the unmodified input, and every metric is
# reported both absolutely and as a delta against it (d_ columns). Reading
# them: dG_separated and delta_unsatHbonds improve going down, sc_value,
# packstat, hbonds_int and dSASA_int going up.
#
# batch_size and number_of_batches stay at 1: raising them fans out again,
# and the extra sequences would need one more MultiplePoseMover to reach the
# CSV.

import csv
from collections import Counter
from pathlib import Path

import pyrosetta
import rosettalink
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from rosettalink.utils import drain_additional_output

rosettalink.init(set_logging_handler="interactive")

# --------------------------------------------------------------------------- #
# Configuration - set these for your installation.
# --------------------------------------------------------------------------- #
HBDESIGNER_REPO = Path("/home/folivieri/HBDesigner")
# Add --no-capture-output after conda run to see HBDesigner output live.
HBDESIGNER_CMD_HEADER = "conda run -n hbdesigner_gpu run_hbdesigner"

LIGANDMPNN_SIF = "/home/folivieri/prosculpt/singularity_files/ligandmpnn.sif"
# Checkpoints as seen from inside the image. All three must be present.
LIGANDMPNN_CKPT = "/app/ligandmpnn/model_params/ligandmpnn_v_32_010_25.pt"
LIGANDMPNN_SC_CKPT = "/app/ligandmpnn/model_params/ligandmpnn_sc_v_32_002_16.pt"

INPUT_PDB = HBDESIGNER_REPO / "examples" / "interface" / "1YRK.pdb"

N_RES = "3"              # residues per network
N_SAMPLES = "50"         # HBDesigner examples use 200
TOP_K = "5"              # networks kept, and so designs produced
N_WORKERS = "8"          # match cpus-per-task in run_hbdesigner.sh
ANCHOR_RES = "B5"        # network anchored on the peptide
OMIT_CHAINS = "B"        # but designed into chain A only
DESIGN_CHAINS = "A"      # chain LigandMPNN redesigns around the network
FIXED_CHAINS = "A"       # one side of the interface for the analyzer

OUTPUT_DIR = Path("hbdesigner_extended_output").resolve()
METRICS_CSV = OUTPUT_DIR / "metrics.csv"

# Declared once and interpolated into both protocols below, so the designs
# and the baseline are measured identically.
INTERFACE_ANALYZER = """<InterfaceAnalyzerMover name="analyze"
                        scorefxn="sfxn"
                        fixedchains="{fixed_chains}"
                        packstat="true"
                        interface_sc="true"
                        pack_separated="true"
                        pack_input="false" />"""

# design_network produces top_k poses; the MultiplePoseMover collects them and
# runs the redesign and the analysis on each.
DESIGN_XML_TEMPLATE = """
<ROSETTASCRIPTS>

    <MOVERS>
        <HBDesigner name="design_network"
            cmd_header="{cmd_header}"
            n_res="{n_res}"
            n_samples="{n_samples}"
            top_k="{top_k}"
            n_workers="{n_workers}"
            anchor_res="{anchor_res}"
            omit_chains="{omit_chains}"
            restore_input_sequence="true"
            reslabel="hbnet"
            prefix_name="HBDesigner_"
            work_dir="{hbdesigner_work_dir}"
            delete_dir="false" />

        <MultiplePoseMover name="redesign_and_analyse">
            <ROSETTASCRIPTS>
                <SCOREFXNS>
                    <ScoreFunction name="sfxn" weights="ref2015" />
                </SCOREFXNS>
                <MOVERS>
                    <LigandMPNN name="redesign"
                        ligandmpnn_path="{ligandmpnn_path}"
                        model_type="ligand_mpnn"
                        checkpoint_ligand_mpnn="{ligandmpnn_ckpt}"
                        fixed_reslabel="hbnet"
                        chains_to_design="{design_chains}"
                        temperature="0.1"
                        batch_size="1"
                        number_of_batches="1"
                        ligand_mpnn_use_side_chain_context="1"
                        pack_side_chains="1"
                        checkpoint_path_sc="{ligandmpnn_sc_ckpt}"
                        number_of_packs_per_design="1"
                        pack_with_ligand_context="1"
                        repack_everything="0"
                        work_dir="{ligandmpnn_work_dir}"
                        delete_dir="false" />

                    {analyzer}
                </MOVERS>
                <PROTOCOLS>
                    <Add mover="redesign" />
                    <Add mover="analyze" />
                </PROTOCOLS>
            </ROSETTASCRIPTS>
        </MultiplePoseMover>
    </MOVERS>

    <PROTOCOLS>
        <Add mover="design_network" />
        <Add mover="redesign_and_analyse" />
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

BASELINE_XML_TEMPLATE = """
<ROSETTASCRIPTS>

    <SCOREFXNS>
        <ScoreFunction name="sfxn" weights="ref2015" />
    </SCOREFXNS>

    <MOVERS>
        {analyzer}
    </MOVERS>

    <PROTOCOLS>
        <Add mover="analyze" />
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

# Score names InterfaceAnalyzerMover writes into the pose.
INTERFACE_METRICS = (
    "dG_separated",
    "dG_separated/dSASAx100",
    "dSASA_int",
    "delta_unsatHbonds",
    "hbonds_int",
    "hbond_E_fraction",
    "sc_value",
    "packstat",
    "nres_int",
)

HBDESIGNER_METRICS = (
    "HBDesigner_Rank",
    "HBDesigner_HB_Score_full",
    "HBDesigner_HB_Score_hb",
    "HBDesigner_Avg_Burial",
    "HBDesigner_saturation",
    "HBDesigner_buried_heavy_unsats",
    "HBDesigner_buried_unsat_Hpol",
)

# --------------------------------------------------------------------------- #


def network_residues(pose):
    """Residues carrying the hbnet label, as chain, number and residue type."""
    selected = ResiduePDBInfoHasLabelSelector("hbnet").apply(pose)
    pdb_info = pose.pdb_info()
    return [
        f"{pdb_info.chain(resnum)}{pdb_info.number(resnum)}{pose.residue(resnum).name3()}"
        for resnum in range(1, pose.total_residue() + 1)
        if selected[resnum]
    ]


def metrics_row(index, cached, network, mutations, pdb_path, baseline_values):
    """One CSV row. baseline_values of None leaves the delta columns empty,
    which is what the baseline row itself gets."""
    row = {"design_index": index, "network": network, "mutations_vs_input": mutations}
    row.update({metric: cached.get(metric) for metric in HBDESIGNER_METRICS})
    for metric in INTERFACE_METRICS:
        value = cached.get(metric)
        row[metric] = value
        reference = baseline_values.get(metric) if baseline_values else None
        row[f"d_{metric}"] = (
            value - reference if value is not None and reference is not None else None
        )
    row["pdb"] = str(pdb_path)
    return row


if not INPUT_PDB.is_file():
    raise SystemExit(f"Input structure not found: {INPUT_PDB}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

analyzer = INTERFACE_ANALYZER.format(fixed_chains=FIXED_CHAINS)

input_pose = pyrosetta.pose_from_file(str(INPUT_PDB))
input_sequence = input_pose.sequence()
chain_sizes = Counter(
    input_pose.pdb_info().chain(resnum)
    for resnum in range(1, input_pose.total_residue() + 1)
)
print(f"\n=== Designing {N_RES}-residue networks across {INPUT_PDB.name} ===")
print(f"Loaded {input_pose.total_residue()} residues, chains {dict(chain_sizes)}")
print(f"anchor_res={ANCHOR_RES}, omit_chains={OMIT_CHAINS}, "
      f"redesigning chain(s) {DESIGN_CHAINS} around each network")

# --- BASELINE --- #
print("\n=== Analysing the input interface ===")
baseline_protocol = XmlObjects.create_from_string(
    BASELINE_XML_TEMPLATE.format(analyzer=analyzer)
).get_mover("ParsedProtocol")
baseline_pose = input_pose.clone()
baseline_protocol.apply(baseline_pose)
baseline = dict(baseline_pose.cache.fast_items())
baseline_pdb = OUTPUT_DIR / "baseline.pdb"
baseline_pose.dump_pdb(str(baseline_pdb))
for metric in INTERFACE_METRICS:
    print(f"\t{metric}: {baseline.get(metric)}")
print(f"\twritten to: {baseline_pdb}")

# First row of the CSV, so every design can be read against it.
rows = [metrics_row("baseline", baseline, "", 0, baseline_pdb, None)]

# --- DESIGN, REDESIGN AND ANALYSE --- #
design_protocol = XmlObjects.create_from_string(
    DESIGN_XML_TEMPLATE.format(
        cmd_header=HBDESIGNER_CMD_HEADER,
        n_res=N_RES,
        n_samples=N_SAMPLES,
        top_k=TOP_K,
        n_workers=N_WORKERS,
        anchor_res=ANCHOR_RES,
        omit_chains=OMIT_CHAINS,
        hbdesigner_work_dir=OUTPUT_DIR / "hbdesigner",
        ligandmpnn_path=LIGANDMPNN_SIF,
        ligandmpnn_ckpt=LIGANDMPNN_CKPT,
        ligandmpnn_sc_ckpt=LIGANDMPNN_SC_CKPT,
        design_chains=DESIGN_CHAINS,
        ligandmpnn_work_dir=OUTPUT_DIR / "ligandmpnn",
        analyzer=analyzer,
    )
).get_mover("ParsedProtocol")

pose = input_pose.clone()
design_protocol.apply(pose)

all_poses = [pose] + drain_additional_output(design_protocol)
print(f"\n=== Analysed {len(all_poses)} design(s) (top_k = {TOP_K}) ===")

for i, design_pose in enumerate(all_poses):
    design_pdb = OUTPUT_DIR / f"design_{i}.pdb"
    design_pose.dump_pdb(str(design_pdb))
    cached = dict(design_pose.cache.fast_items())

    missing = [m for m in INTERFACE_METRICS if m not in cached]
    if missing:
        print(f"WARNING: design {i} is missing interface metric(s) {missing}")

    sequence = design_pose.sequence()
    mutations = sum(
        1 for a, b in zip(input_sequence, sequence) if a != b
    ) if len(sequence) == len(input_sequence) else None

    row = metrics_row(
        i, cached, " ".join(network_residues(design_pose)), mutations, design_pdb, baseline
    )
    rows.append(row)

    print(f"\nDesign {i}: {row['network'] or 'no hbnet residues labelled'}")
    print(f"\t{mutations} position(s) differ from the input")
    print(f"\tHBDesigner: HB_Score_full={row['HBDesigner_HB_Score_full']}, "
          f"saturation={row['HBDesigner_saturation']}, "
          f"buried_unsats={row['HBDesigner_buried_heavy_unsats']}")
    print(f"\tdG_separated={row['dG_separated']} (d {row['d_dG_separated']})")
    print(f"\tdelta_unsatHbonds={row['delta_unsatHbonds']} (d {row['d_delta_unsatHbonds']})")
    print(f"\thbonds_int={row['hbonds_int']} (d {row['d_hbonds_int']}), "
          f"sc_value={row['sc_value']}, packstat={row['packstat']}")
    print(f"\twritten to: {design_pdb}")

with open(METRICS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n=== Metrics CSV written to {METRICS_CSV} ===")
