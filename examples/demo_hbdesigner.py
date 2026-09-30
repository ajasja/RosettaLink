# @file examples/demo_hbdesigner.py
# @brief Designs buried hydrogen bond networks across the 1YRK interface with
#        the HBDesigner mover, then reports each network and its scores.
#
# Mirrors HBDesigner own examples/interface/one_sided/run_hbdes.sh, with
# n_samples lowered for a quicker first run. 1YRK is a 126-residue domain
# (chain A) with a 12-residue peptide bound (chain B). anchor_res anchors the
# network on the peptide, omit_chains keeps design itself on chain A, so the
# networks are built into the domain around a fixed peptide contact.
#
# top_k networks come back per apply(): the best in the pose, the rest
# through get_additional_output(). Residues forming each network carry the
# hbnet label, so they can be selected downstream.
#
# The other interface examples map onto the same attributes:
#   two_sided     drop anchor_res and omit_chains, design both sides
#   symm_lazy     symm_chains="A,B" on a homodimer
#   symm_strict   symm_chains="A,B" plus symm_file="5J0K.symm"

import pyrosetta
import rosettalink
from collections import Counter
from pathlib import Path
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from rosettalink.utils import drain_additional_output

rosettalink.init(set_logging_handler="interactive")

# --------------------------------------------------------------------------- #
# Configuration - set these for your installation.
# --------------------------------------------------------------------------- #
HBDESIGNER_REPO = Path("/home/folivieri/HBDesigner")
# Command reaching the run_hbdesigner entry point in its own environment.
# Add --no-capture-output after conda run to see HBDesigner output live.
HBDESIGNER_CMD_HEADER = "conda run -n hbdesigner_gpu run_hbdesigner"

INPUT_PDB = HBDESIGNER_REPO / "examples" / "interface" / "1YRK.pdb"

N_RES = "3"            # residues per network
N_SAMPLES = "50"       # HBDesigner examples use 200
TOP_K = "5"            # networks kept, and so poses produced
N_WORKERS = "8"        # match cpus-per-task in run_hbdesigner.sh
ANCHOR_RES = "B5"      # network anchored on the peptide
OMIT_CHAINS = "B"      # but designed into chain A only

OUTPUT_DIR = Path("hbdesigner_output_restore_input").resolve()

XML_TEMPLATE = """
<ROSETTASCRIPTS>

    <MOVERS>
        <HBDesigner name="design_network"
            cmd_header="{cmd_header}"
            n_res="{n_res}"
            n_samples="{n_samples}"
            top_k="{top_k}"
            n_workers="{n_workers}"
            anchor_res="{anchor_res}"
            restore_input_sequence="true"
            omit_chains="{omit_chains}"
            reslabel="hbnet"
            prefix_name="HBDesigner_"
            work_dir="{work_dir}"
            delete_dir="false" />
    </MOVERS>

    <PROTOCOLS>
        <Add mover="design_network" />
    </PROTOCOLS>

</ROSETTASCRIPTS>
"""

SCORE_KEYS = (
    "HBDesigner_Rank",
    "HBDesigner_HB_Score_full",
    "HBDesigner_HB_Score_hb",
    "HBDesigner_Avg_Burial",
    "HBDesigner_saturation",
    "HBDesigner_buried_heavy_unsats",
    "HBDesigner_buried_unsat_Hpol",
)

# --------------------------------------------------------------------------- #

if not INPUT_PDB.is_file():
    raise SystemExit(f"Input structure not found: {INPUT_PDB}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

xml = XmlObjects.create_from_string(
    XML_TEMPLATE.format(
        cmd_header=HBDESIGNER_CMD_HEADER,
        n_res=N_RES,
        n_samples=N_SAMPLES,
        top_k=TOP_K,
        n_workers=N_WORKERS,
        anchor_res=ANCHOR_RES,
        omit_chains=OMIT_CHAINS,
        work_dir=OUTPUT_DIR / "hbdesigner",
    )
)
protocol = xml.get_mover("ParsedProtocol")

pose = pyrosetta.pose_from_file(str(INPUT_PDB))
chain_sizes = Counter(
    pose.pdb_info().chain(resnum) for resnum in range(1, pose.total_residue() + 1)
)
print(f"\n=== Designing {N_RES}-residue networks across {INPUT_PDB.name} ===")
print(f"Loaded {pose.total_residue()} residues, chains {dict(chain_sizes)}")
print(f"anchor_res={ANCHOR_RES}, omit_chains={OMIT_CHAINS}")

protocol.apply(pose)

all_poses = [pose] + drain_additional_output(protocol)
print(f"Produced {len(all_poses)} network(s) (top_k = {TOP_K})")

hbnet_selector = ResiduePDBInfoHasLabelSelector("hbnet")

for i, network_pose in enumerate(all_poses):
    out_pdb = OUTPUT_DIR / f"hbnet_{i}.pdb"
    network_pose.dump_pdb(str(out_pdb))

    cached = dict(network_pose.cache.fast_items())
    pdb_info = network_pose.pdb_info()
    selected = hbnet_selector.apply(network_pose)
    network = [
        (pdb_info.chain(resnum), pdb_info.number(resnum), network_pose.residue(resnum).name3())
        for resnum in range(1, network_pose.total_residue() + 1)
        if selected[resnum]
    ]

    residues = ", ".join(f"{chain}{number}{name}" for chain, number, name in network)
    print(f"\nNetwork {i}: {residues or 'no hbnet residues labelled'}")

    # omit_chains should keep every designed residue off chain B.
    on_omitted = [f"{chain}{number}" for chain, number, _ in network if chain in OMIT_CHAINS]
    if on_omitted:
        print(f"\tWARNING: residues on omitted chain(s): {on_omitted}")

    for key in SCORE_KEYS:
        print(f"\t{key}: {cached.get(key)}")
    print(f"\twritten to: {out_pdb}")
