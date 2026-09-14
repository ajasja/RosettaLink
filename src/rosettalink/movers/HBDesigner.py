# @file movers/HBDesigner.py
# @brief Rosetta mover to run HBDesigner, which designs buried hydrogen bond
#        networks into an input backbone.
#
# HBDesigner (https://github.com/RosettaCommons/HBDesigner) is a command line
# tool, so cmd_header is the full command prefix that reaches its entry
# point, e.g. "conda run -n hbdesigner_gpu run_hbdesigner" or
# "pixi run --manifest-path=/path/to/HBDesigner/pyproject.toml run_hbdesigner".
# It pins python to 3.10 and brings its own torch, so it belongs in its own
# environment.
#
# The pose is written to input.pdb and HBDesigner is run over it. top_k
# networks come back ranked; the best enters the pose and the rest are
# exposed through get_additional_output().
#
# Every attribute in OPTION_HELP is passed through as --name value only when
# set, so HBDesigner own defaults apply otherwise. Values are quoted, so
# option values containing commas or pipes (guide_seq="S,N|Q,T") are safe.
# Anything not listed can be reached through extra_args.
#
# Residues forming the designed network are labelled with reslabel ("hbnet"
# by default), so downstream protocols can select them with
# ResiduePDBInfoHasLabelSelector. Labels on the input pose are carried over.
#
# HBDesigner returns glycine at every position outside the network.
# restore_input_sequence replaces those positions with the residues of the
# input pose, giving the original structure carrying only the designed
# network, with all coordinates preserved.
#
# Scores from the HBDesigner stats csv (HB_Score_full, HB_Score_hb,
# Avg_Burial, saturation, buried_heavy_unsats, buried_unsat_Hpol, Rank) are
# attached to each pose under prefix_name.

import csv
import os
import re
import shutil
import tempfile
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.pose import setPoseExtraScore

from rosettalink.decorators import register_mover
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer

# Attributes passed straight through to the run_hbdesigner command line as
# --name value, each only when set. Descriptions name the HBDesigner default.
OPTION_HELP = {
    "design_model": "Design model to use. HBDesigner default is design_020 (moderate noise)",
    "n_res": "Size of the desired network, in residues. HBDesigner default is 2",
    "n_samples": "Number of unique samples to generate before packing and scoring. HBDesigner default is 100",
    "top_k": "How many ranked networks to keep. HBDesigner default is 5. The best enters the pose, the rest come back through get_additional_output()",
    "n_workers": "Workers for parallelisation during packing. HBDesigner default is 1",
    "min_burial": "Minimum burial for designable positions. HBDesigner default is 0.0",
    "min_core_res": "Minimum number of core residues required per network. HBDesigner default is 0",
    "min_sat": "Minimum saturation score allowed in the network. HBDesigner default is 0.5",
    "max_BUNs": "Maximum buried unsatisfied heavy atoms allowed. HBDesigner default is 0",
    "max_BUPHs": "Maximum buried unsatisfied polar hydrogens allowed. HBDesigner default is 5",
    "max_hb_energy": "Maximum hydrogen bond energy allowed in the network. HBDesigner default is 0.0",
    "max_hb_score": "Maximum energy score allowed for returned networks. HBDesigner default is 0.0",
    "guide_res": "Residues to build the network around, in PDB chain and resnum format, e.g. A12,B13",
    "guide_radius": "Angstrom cutoff from the guide atom. Off by default",
    "guide_seq": "Guide sequence, e.g. S,N,T for a full specification, X,T,X for a partial one, or S,N|Q,T for ambiguity",
    "anchor_res": "Residues to use as anchors during design, in PDB chain and resnum format",
    "omit_AA": "Amino acids to omit from design, e.g. R,K",
    "omit_chains": "Chains to omit from design",
    "sel_chains": "Chains to run HBDesigner on, e.g. A,C. All chains by default",
    "symm_chains": "Chains to symmetrise output networks over",
    "symm_file": "Symmetry file (.symm) for strict symmetry",
    "seed": "Random seed for sampling. No seed by default",
}

# Stats csv columns worth attaching to the pose as scores. The others
# (Scaffold, Output_PDB, network) are text.
SCORE_COLUMNS = (
    "Rank",
    "HB_Score_full",
    "HB_Score_hb",
    "Avg_Burial",
    "saturation",
    "buried_heavy_unsats",
    "buried_unsat_Hpol",
)

# Entries of the network column, e.g. A12S:A26T -> chain A resnum 12, etc.
NETWORK_RESIDUE = re.compile(r"^([A-Za-z])(-?\d+)([A-Z])$")


def parse_network(network):
    """Turns a network string such as A12S:A26T into [(chain, resnum), ...].
    Unrecognised entries are skipped."""
    residues = []
    for entry in network.split(":"):
        entry = entry.strip()
        if not entry:
            continue
        match = NETWORK_RESIDUE.match(entry)
        if match:
            residues.append((match.group(1), int(match.group(2))))
    return residues


class HBDesigner(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(
        self,
        cmd_header=None,
        cpu="0",
        restore_input_sequence="0",
        reslabel="hbnet",
        prefix_name="HBDesigner_",
        extra_args=None,
        work_dir=None,
        delete_dir=None,
        **options
    ):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)

        self.cmd_header_ = cmd_header
        self.cpu_ = cpu
        self.restore_input_sequence_ = restore_input_sequence
        self.reslabel_ = reslabel
        self.prefix_name_ = prefix_name
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir
        # Pass-through command line options, empty meaning "leave unset".
        self.options_ = {name: options.get(name, "") for name in OPTION_HELP}
        # Populated by apply(): every network beyond the best one, exposed
        # via the standard Mover::get_additional_output() mechanism - same
        # one-to-many pattern as RFDiffusion/LigandMPNN.
        self.additional_poses_ = []

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[HBDesigner]")

        self.tracer_info << (
            f"Initializing HBDesigner:\n"
            f"\tcmd_header: {self.cmd_header_}\n"
            f"\tcpu: {self.cpu_}\n"
            f"\trestore_input_sequence: {self.restore_input_sequence_}\n"
            f"\treslabel: {self.reslabel_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()

    def clone(self):
        copy = HBDesigner()
        copy.cmd_header_ = self.cmd_header_
        copy.cpu_ = self.cpu_
        copy.restore_input_sequence_ = self.restore_input_sequence_
        copy.reslabel_ = self.reslabel_
        copy.prefix_name_ = self.prefix_name_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        copy.options_ = dict(self.options_)
        HBDesigner.clones_.append(copy)
        return copy

    def apply(self, pose):
        # --- working directory --- #
        # Fresh, self-cleaning temp directory every call (or a fresh
        # subdirectory of a configured work_dir) - never stored back onto
        # self.work_dir_, so this mover instance can safely be applied more
        # than once (e.g. from a RosettaScripts MultiplePoseMover).
        if self.work_dir_ is None or self.work_dir_ == "":
            temp_dir = tempfile.TemporaryDirectory()
            run_dir = Path(temp_dir.name)
            self.tracer_info << f"No working directory specified, using temporary directory: {run_dir}\n" and self.tracer_info.flush()
        else:
            temp_dir = None
            os.makedirs(self.work_dir_, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(dir=self.work_dir_))

        # Keep a copy of the pose as handed to us: source of reslabels to
        # carry forward, since the HBDesigner output pdb carries none.
        input_pose = pose.clone()

        # --- INPUT --- #
        # A fixed input name keeps the output names predictable: HBDesigner
        # prefixes them with the input file stem.
        input_pdb = run_dir / "input.pdb"
        pose.dump_pdb(str(input_pdb))
        self.tracer_info << f"Wrote input pose to {input_pdb}\n" and self.tracer_info.flush()

        # --- RUN HBDESIGNER --- #
        hbdesigner_cmd_str = f"{self.cmd_header_} --pdb {input_pdb} --out_dir {run_dir}"
        for name, value in self.options_.items():
            if value != "" and value is not None:
                # Quoted: guide_seq values can contain a pipe.
                hbdesigner_cmd_str += f' --{name} "{value}"'
        if self.cpu_:
            hbdesigner_cmd_str += " --cpu"
        if self.extra_args_:
            hbdesigner_cmd_str += f" {self.extra_args_}"

        self.tracer_info << f"Running HBDesigner: {hbdesigner_cmd_str}\n" and self.tracer_info.flush()
        run_and_log(hbdesigner_cmd_str, self.tracer_info, self.tracer_error)

        # --- OUTPUT NETWORKS --- #
        stats = self._read_stats(run_dir / "input_HBDes_stats.csv")

        designed_poses = []
        rank = 1
        while True:
            design_pdb = run_dir / f"input_HBDes_rank_{rank}.pdb"
            if not design_pdb.is_file():
                break
            designed_poses.append(
                self._load_and_label_design(design_pdb, stats.get(str(rank), {}), input_pose)
            )
            rank += 1

        if not designed_poses:
            self.tracer_error << (
                f"No HBDesigner output pdb files found under {run_dir}. HBDesigner returns "
                f"nothing when no network satisfies the requested constraints; try raising "
                f"n_samples or relaxing max_BUNs/max_BUPHs/min_sat.\n"
            ) and self.tracer_error.flush()
            raise RuntimeError(f"No HBDesigner output pdb files found under {run_dir}")
        self.tracer_info << f"HBDesigner returned {len(designed_poses)} network(s)\n" and self.tracer_info.flush()

        # Primary output: same convention as the other movers - the pose the
        # caller already holds a reference to gets the top-ranked network.
        pose.assign(designed_poses[0])

        # Every further network is handed off through
        # Mover::get_additional_output(), the standard Rosetta one-to-many
        # mechanism, instead of being silently discarded.
        self.additional_poses_ = list(designed_poses[1:])

        # --- CLEANUP --- #
        try:
            if temp_dir:
                temp_dir.cleanup()
                self.tracer_info << f"Cleaned up temporary directory: {run_dir}\n" and self.tracer_info.flush()
            elif self.delete_dir_:
                shutil.rmtree(run_dir)
                self.tracer_info << f"Deleted run directory: {run_dir}\n" and self.tracer_info.flush()
        except Exception:
            self.tracer_error << f"Failed to clean up run directory: {run_dir}\n" and self.tracer_error.flush()

    def _read_stats(self, stats_csv):
        """Reads the HBDesigner stats csv into {rank: row}. Returns an empty
        dict when the file is absent, so a run still yields poses."""
        if not stats_csv.is_file():
            self.tracer_warning << f"No HBDesigner stats csv found at {stats_csv}\n" and self.tracer_warning.flush()
            return {}
        with open(stats_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        return {row["Rank"]: row for row in rows if row.get("Rank")}

    def _restore_input_sequence(self, design_pose, input_pose, network_resnums):
        """Returns design_pose with every residue outside network_resnums
        replaced by the residue of input_pose at the same PDB chain and
        number, coordinates included. Positions with no match in input_pose
        are left as they are and reported."""
        restored = design_pose.clone()
        pdb_info_design = restored.pdb_info()
        pdb_info_input = input_pose.pdb_info()
        if pdb_info_design is None or pdb_info_input is None:
            self.tracer_warning << (
                "Cannot restore the input sequence without pdb_info on both poses\n"
            ) and self.tracer_warning.flush()
            return restored

        replaced = 0
        missing = []
        for resnum in range(1, restored.total_residue() + 1):
            if resnum in network_resnums:
                continue
            source = pdb_info_input.pdb2pose(
                pdb_info_design.chain(resnum), pdb_info_design.number(resnum)
            )
            if source == 0:
                missing.append(f"{pdb_info_design.chain(resnum)}{pdb_info_design.number(resnum)}")
                continue
            restored.replace_residue(resnum, input_pose.residue(source), False)
            replaced += 1

        if missing:
            self.tracer_warning << (
                f"Left as glycine, no matching residue in the input pose: {missing}\n"
            ) and self.tracer_warning.flush()
        self.tracer_info << (
            f"\trestored {replaced} input residue(s) around a network of "
            f"{len(network_resnums)}\n"
        ) and self.tracer_info.flush()
        return restored

    def _load_and_label_design(self, design_pdb, stats_row, input_pose):
        """Load one designed network, carry over reslabels from the input
        pose, label the network residues, and attach the scores of its stats
        row."""
        pose = pyrosetta.pose_from_file(str(design_pdb))

        pdb_info_old = input_pose.pdb_info()
        pdb_info_new = pose.pdb_info()
        if pdb_info_old is not None and pdb_info_new is not None:
            for resnum in range(1, min(input_pose.total_residue(), pose.total_residue()) + 1):
                for reslabel in pdb_info_old.get_reslabels(resnum):
                    pdb_info_new.add_reslabel(resnum, reslabel)

        # --- NETWORK RESIDUE LABELS --- #
        network = stats_row.get("network", "")
        network_resnums = []
        if network and pdb_info_new is not None:
            for chain, pdb_resnum in parse_network(network):
                resnum = pdb_info_new.pdb2pose(chain, pdb_resnum)
                if resnum == 0:
                    self.tracer_warning << (
                        f"Network residue {chain}{pdb_resnum} not found in {design_pdb.name}\n"
                    ) and self.tracer_warning.flush()
                    continue
                pdb_info_new.add_reslabel(resnum, self.reslabel_)
                network_resnums.append(resnum)
            self.tracer_info << (
                f"\tnetwork {network} labelled {self.reslabel_} at {network_resnums}\n"
            ) and self.tracer_info.flush()

        # --- INPUT SEQUENCE --- #
        if self.restore_input_sequence_:
            pose = self._restore_input_sequence(pose, input_pose, set(network_resnums))

        # --- SCORES --- #
        for column in SCORE_COLUMNS:
            value = stats_row.get(column)
            if value in (None, ""):
                continue
            try:
                score_value = float(value)
            except ValueError:
                continue
            score_name = f"{self.prefix_name_}{column}"
            setPoseExtraScore(pose, score_name, score_value)
            self.tracer_info << f"\t{score_name}: {score_value}\n" and self.tracer_info.flush()

        return pose

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one additional network per
        # call, None once exhausted.
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ HBDesigner...\n\tself: {self}\n\ttag: {tag}\n\tdata: {data}\n" and self.tracer_debug.flush()

        # required attributes
        self.cmd_header_ = tag.get_option_string("cmd_header")

        # optional attributes
        self.cpu_ = tag.get_option_bool("cpu") if tag.hasOption("cpu") else False
        self.restore_input_sequence_ = tag.get_option_bool("restore_input_sequence") if tag.hasOption("restore_input_sequence") else False
        self.reslabel_ = tag.get_option_string("reslabel") if tag.hasOption("reslabel") else "hbnet"
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else "HBDesigner_"
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""
        self.delete_dir_ = tag.get_option_bool("delete_dir") if tag.hasOption("delete_dir") else False

        self.options_ = {
            name: (tag.get_option_string(name) if tag.hasOption(name) else "")
            for name in OPTION_HELP
        }

        set_options = {name: value for name, value in self.options_.items() if value != ""}
        self.tracer_info << (
            f"Parsed options:\n"
            f"\tcmd_header: {self.cmd_header_}\n"
            f"\tcpu: {self.cpu_}\n"
            f"\trestore_input_sequence: {self.restore_input_sequence_}\n"
            f"\treslabel: {self.reslabel_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
            f"\tHBDesigner options: {set_options}\n"
        ) and self.tracer_info.flush()

    @staticmethod
    def mover_name():
        return "HBDesigner"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_boolean

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        attrlist.append(XMLSchemaAttribute.required_attribute(
            "cmd_header",
            XMLSchemaType(xs_string),
            "Start of the command string, reaching the HBDesigner entry point. For example conda run -n ENVNAME run_hbdesigner, or pixi run --manifest-path=/path/to/HBDesigner/pyproject.toml run_hbdesigner"))

        for name, help_text in OPTION_HELP.items():
            attrlist.append(XMLSchemaAttribute.attribute_w_default(
                name,
                XMLSchemaType(xs_string),
                f"{help_text}. Passed through only when set",
                ""))

        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "cpu",
            XMLSchemaType(xs_boolean),
            "Whether to run on cpu instead of gpu",
            "false"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "restore_input_sequence",
            XMLSchemaType(xs_boolean),
            "Whether to replace every position outside the designed network with the residue of the input pose. HBDesigner returns glycine at those positions, so this yields the original structure carrying only the network",
            "false"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "reslabel",
            XMLSchemaType(xs_string),
            "Residue label applied to the residues forming the designed network",
            "hbnet"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "prefix_name",
            XMLSchemaType(xs_string),
            "Prefix to all metrics reported by the mover",
            "HBDesigner_"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for the run_hbdesigner command, for any option this mover does not expose",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Parent directory under which each apply call gets its own fresh subdirectory (so this mover instance can safely be applied more than once, e.g. from a RosettaScripts MultiplePoseMover). If not provided, a new temporary directory is used per call instead. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete this call run subdirectory after every network has been read from disk into the primary pose or the additional-output poses",
            "false"))

        description = "Runs HBDesigner to design buried hydrogen bond networks into the pose backbone."

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd, cls.mover_name(), description, attrlist)


@register_mover
class HBDesignerCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = HBDesigner()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return HBDesigner.mover_name()

    def provide_xml_schema(self, xsd):
        HBDesigner.provide_xml_schema(xsd)
