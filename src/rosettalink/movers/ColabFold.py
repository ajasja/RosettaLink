# @file movers/ColabFold.py
# @brief Rosetta mover to run ColabFold

"""
ColabFold command strings should match those in do_cycling.
    reference: https://github.com/ajasja/prosculpt/blob/5211fe061fe0cf03f79a9912fcb9c0f96fc11875/rfdiff_mpnn_af2_merged.py
All scoring metrics should match those in rename_pdb_create_csv_colabfold.
    reference: https://github.com/ajasja/prosculpt/blob/main/prosculpt.py
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector
from pyrosetta.rosetta.core.pose import setPoseExtraScore

from rosettalink.decorators import register_mover
from rosettalink.utils import parse_fasta_records
from rosettalink.utils import resolve_rmsd_atoms
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer


class ColabFold(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(
        self, 
        replace_pose="1", 
        fasta="",
        models="1,2,3,4,5", 
        msa_mode="single_sequence", 
        rank="auto", 
        prefix_name="AF2_",
        cmd_header=None,
        extra_args=None, 
        work_dir=None, 
        delete_dir="0"
    ):        
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.rmsd_metrics = []
        # Populated by apply() when folding a fasta: one pose per record
        # beyond the first, exposed through get_additional_output().
        self.additional_poses_ = []
        
        self.replace_pose_ = replace_pose
        self.fasta_ = fasta
        self.models_ = models
        self.msa_mode_ = msa_mode
        self.rank_ = rank
        self.prefix_name_ = prefix_name
        self.cmd_header_ = cmd_header
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[ColabFold]")

        self.tracer_info << (
            f"Initializing ColabFold:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\tmodels: {self.models_}\n"
            f"\tmsa_mode: {self.msa_mode_}\n"
            f"\trank: {self.rank_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\tcmd_header: {self.cmd_header_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()
        
    def clone(self):
        copy = ColabFold()
        copy.replace_pose_ = self.replace_pose_
        copy.fasta_ = self.fasta_
        copy.models_ = self.models_
        copy.msa_mode_ = self.msa_mode_
        copy.rank_ = self.rank_
        copy.prefix_name_ = self.prefix_name_
        copy.cmd_header_ = self.cmd_header_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        copy.rmsd_metrics = self.rmsd_metrics.copy()
        ColabFold.clones_.append(copy)
        return copy
    
    def apply(self, pose):
        # --- working directory --- #
        # Fresh, self-cleaning temp directory every call (or a fresh
        # subdirectory of a configured work_dir) - never stored back onto
        # self.work_dir_, so this mover instance can safely be applied more
        # than once (e.g. from a RosettaScripts MultiplePoseMover).
        if self.work_dir_ is None or self.work_dir_ == "":
            temp_dir = tempfile.TemporaryDirectory()
            work_dir = Path(temp_dir.name)
            self.tracer_info << f"No working directory specified, using temporary directory: {work_dir}\n" and self.tracer_info.flush()
        else:
            temp_dir = None
            os.makedirs(self.work_dir_, exist_ok=True)
            work_dir = Path(tempfile.mkdtemp(dir=self.work_dir_))

        # --- INPUT FASTA --- #
        # With fasta set the records come from that file, one per pose, with
        # the chains of a record colon separated - which is also how
        # colabfold_batch itself denotes a complex. Otherwise the pose
        # sequence is written as the single record "input".
        if self.fasta_:
            records = parse_fasta_records(self.fasta_)
            self.tracer_info << (
                f"Folding {len(records)} record(s) from {self.fasta_}\n"
            ) and self.tracer_info.flush()
        else:
            records = [("input", [pose.sequence()])]

        fasta_path = work_dir / "input.fasta"
        with open(fasta_path, "w") as f:
            for record_id, chains in records:
                f.write(f">{record_id}\n{':'.join(chains)}\n")
        self.tracer_info << f"Writing {len(records)} record(s): {fasta_path}\n" and self.tracer_info.flush()

        # --- RUN COLABFOLD --- #
        colabfold_cmd_str = (
            f"{self.cmd_header_} "
            f"--model-order {self.models_} "
            f"--msa-mode {self.msa_mode_} "
            f"--rank {self.rank_} "
            f"{self.extra_args_ if self.extra_args_ else ''} "
            f"{fasta_path} "
            f"{work_dir}"
        )

        self.tracer_info << f"Running ColabFold: {colabfold_cmd_str}\n" and self.tracer_info.flush()
        run_and_log(colabfold_cmd_str, self.tracer_info, self.tracer_error)

        # --- ONE POSE PER FASTA RECORD --- #
        # Structures folded from a fasta have no correspondence to the input
        # pose, so its labels are not carried over and no RMSD is computed.
        if self.fasta_:
            if self.rmsd_metrics:
                self.tracer_warning << (
                    "Skipping RMSD metrics: structures folded from a fasta have no "
                    "correspondence to the input pose\n"
                ) and self.tracer_warning.flush()

            record_poses = []
            for record_id, _ in records:
                matches = sorted(work_dir.glob(f"{record_id}_*rank_001*.pdb"))
                if not matches:
                    matches = sorted(work_dir.glob(f"{record_id}_*.pdb"))
                if not matches:
                    self.tracer_error << (
                        f"No ColabFold output pdb found for record {record_id} in {work_dir}\n"
                    ) and self.tracer_error.flush()
                    raise RuntimeError(
                        f"No ColabFold output pdb found for record {record_id} in {work_dir}"
                    )
                record_pose = pyrosetta.pose_from_file(str(matches[0]))
                pyrosetta.rosetta.core.pose.add_comment(record_pose, "fasta_id", record_id)
                self._attach_scores(record_pose, work_dir, record_id)
                self.tracer_info << (
                    f"{record_id}: {matches[0]}\n"
                ) and self.tracer_info.flush()
                record_poses.append(record_pose)

            pose.assign(record_poses[0])
            self.additional_poses_ = list(record_poses[1:])
            self._cleanup(temp_dir, work_dir)
            return

        # --- OUTPUT PDB --- #
        pdb_files = sorted(work_dir.glob("*.pdb"))
        if not pdb_files:
            self.tracer_error << f"No PDB files found in output directory: {work_dir}\n" and self.tracer_error.flush()
            raise RuntimeError(f"No PDB files found in output directory: {work_dir}")
        self.tracer_info << f"PDB files found: {[str(pdb) for pdb in pdb_files]}\n" and self.tracer_info.flush()

        # --- BEST PREDICTION --- #
        best_pdb = next((pdb for pdb in pdb_files if "rank_001" in pdb.name), pdb_files[0])
        self.tracer_info << f"Best ColabFold prediction: {best_pdb}\n" and self.tracer_info.flush()

        # --- REPLACE POSE --- #
        input_pose = pose.clone()
        if self.replace_pose_:
            best_pose = pyrosetta.pose_from_file(str(best_pdb))
            pose.assign(best_pose)
            
            reslabels_all = {}
            pdb_info_old = input_pose.pdb_info()
            for resnum in range(1, input_pose.total_residue() + 1):
                reslabels = pdb_info_old.get_reslabels(resnum)
                if reslabels:
                    reslabels_all[resnum] = reslabels

            pdb_info_new = pose.pdb_info()
            for resnum, reslabels in reslabels_all.items():
                for reslabel in reslabels:
                    pdb_info_new.add_reslabel(resnum, reslabel)

            self.tracer_info << f"Replaced pose with best ColabFold prediction: {best_pdb}\n" and self.tracer_info.flush()

        # --- STORE SCORES --- #
        json_files = sorted(work_dir.glob("*scores*.json"))
        if not json_files:
            self.tracer_warning << f"No JSON files with scores found in output directory: {work_dir}\n" and self.tracer_warning.flush()
        else:
            best_json = next((j for j in json_files if "rank_001" in j.name), json_files[0])
            self.tracer_info << f"Extracting scores from {best_json}\n" and self.tracer_info.flush()

            with open(best_json, "r") as f:
                scores = json.load(f)

                if "plddt" in scores:
                    # plddt_all
                    plddt = float(np.mean(scores["plddt"]))
                    setPoseExtraScore(pose, f"{self.prefix_name_}plddt", plddt)
                    self.tracer_info << f"\t{self.prefix_name_}plddt: {plddt}\n" and self.tracer_info.flush()

                    # plddt_sculpted
                    pdb_info = pose.pdb_info()
                    sculpted_residues = set(
                        resnum - 1
                        for resnum in range(1, pose.total_residue() + 1)
                        if pdb_info.res_haslabel(resnum, "sculpted")
                    )
                    plddt_per_residue = [scores["plddt"][i] for i in range(len(scores["plddt"])) if i in sculpted_residues]
                    plddt_sculpted = float(np.mean(plddt_per_residue))
                    setPoseExtraScore(pose, f"{self.prefix_name_}plddt_sculpted", plddt_sculpted)
                    self.tracer_info << f"\t{self.prefix_name_}plddt_sculpted: {plddt_sculpted}\n" and self.tracer_info.flush()  

                if "pae" in scores:
                    # pae_all
                    pae = float(np.mean(scores["pae"]))
                    setPoseExtraScore(pose, f"{self.prefix_name_}pae", pae)
                    self.tracer_info << f"\t{self.prefix_name_}pae: {pae}\n" and self.tracer_info.flush()
        
        # --- RMSD METRICS --- #
        for rmsd in self.rmsd_metrics:
            # reslabel_input selects on the pose the mover was handed,
            # reslabel_prediction on the prediction.
            residue_selector_input = ResiduePDBInfoHasLabelSelector(rmsd["reslabel_input"])
            residue_selector_prediction = ResiduePDBInfoHasLabelSelector(rmsd["reslabel_prediction"])

            # reslabel_superimpose, when set, superimposes over a different
            # set of residues than the RMSD is measured over - e.g. align on
            # a fixed target chain and measure a designed binder, which
            # reports placement rather than fold alone. Defaults to the
            # measured residues.
            superimpose_label = rmsd.get("reslabel_superimpose", "")
            residue_selector_super = (
                ResiduePDBInfoHasLabelSelector(superimpose_label) if superimpose_label else None
            )

            # A label that matches no residue (e.g. motif on a design with no
            # scaffolded motif) would otherwise be an RMSD over nothing.
            # Skip it and leave the score unset rather than fail the run.
            selections = [
                (rmsd["reslabel_input"], residue_selector_input, input_pose),
                (rmsd["reslabel_prediction"], residue_selector_prediction, pose),
            ]
            if residue_selector_super is not None:
                selections.append((superimpose_label, residue_selector_super, pose))
            empty_labels = [
                label for label, selector, target in selections if sum(selector.apply(target)) == 0
            ]
            if empty_labels:
                self.tracer_warning << (
                    f"Skipping RMSD {rmsd['name']}: label(s) {empty_labels} select no residues\n"
                ) and self.tracer_warning.flush()
                continue

            rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
            rmsd_metric.set_residue_selector(residue_selector_prediction)
            rmsd_metric.set_residue_selector_reference(residue_selector_input)
            rmsd_metric.set_comparison_pose(input_pose)
            if residue_selector_super is not None:
                rmsd_metric.set_residue_selector_super(residue_selector_super)
            # RMSDMetric does not align by default, which for a prediction in
            # its own reference frame measures the frame offset rather than
            # any structural difference.
            rmsd_metric.set_run_superimpose(True)
            # Governs the superposition as well as the measurement. Rosetta
            # defaults to all heavy atoms, i.e. sidechains included.
            rmsd_metric.set_rmsd_type(resolve_rmsd_atoms(rmsd["atoms"]))

            rmsd_value = rmsd_metric.calculate(pose)
            rmsd_name = f"{self.prefix_name_}{rmsd['name']}"
            setPoseExtraScore(pose, rmsd_name, rmsd_value)
            self.tracer_info << f"\t{rmsd_name}: {rmsd_value}\n" and self.tracer_info.flush()

        # --- CLEANUP --- #
        self._cleanup(temp_dir, work_dir)

    def _attach_scores(self, pose, work_dir, record_id):
        """Attaches the mean pLDDT and PAE of one fasta record, read from the
        rank_001 scores json ColabFold writes for it."""
        json_files = sorted(work_dir.glob(f"{record_id}_*scores*rank_001*.json"))
        if not json_files:
            json_files = sorted(work_dir.glob(f"{record_id}_*scores*.json"))
        if not json_files:
            self.tracer_warning << (
                f"No scores json found for record {record_id} in {work_dir}\n"
            ) and self.tracer_warning.flush()
            return

        with open(json_files[0], "r") as f:
            scores = json.load(f)
        for key in ("plddt", "pae"):
            if key in scores:
                value = float(np.mean(scores[key]))
                score_name = f"{self.prefix_name_}{key}"
                setPoseExtraScore(pose, score_name, value)
                self.tracer_info << f"\t{record_id} {score_name}: {value}\n" and self.tracer_info.flush()

    def _cleanup(self, temp_dir, work_dir):
        """Removes this call working directory, when it is a temporary one or
        delete_dir asks for it."""
        if temp_dir:
            temp_dir.cleanup()
            self.tracer_info << f"Cleaned up temporary directory: {work_dir}\n" and self.tracer_info.flush()
        elif self.delete_dir_:
            try:
                shutil.rmtree(work_dir)
                self.tracer_info << f"Deleted working directory: {work_dir}\n" and self.tracer_info.flush()
            except Exception:
                self.tracer_error << f"Failed to delete working directory: {work_dir}\n" and self.tracer_error.flush()

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one additional pose per call,
        # None once exhausted. Only ever populated when folding a fasta.
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ ColabFold...\n\tself: {self}\n\ttag: {tag}\n\tdata: {data}\n" and self.tracer_debug.flush()

        # required attributes
        self.cmd_header_ = tag.get_option_string("cmd_header")

        # optional attributes
        self.fasta_ = tag.get_option_string("fasta") if tag.hasOption("fasta") else ""
        self.replace_pose_ = tag.get_option_bool("replace_pose") if tag.hasOption("replace_pose") else "1"
        self.models_ = tag.get_option_string("models") if tag.hasOption("models") else "1,2,3,4,5"
        self.msa_mode_ = tag.get_option_string("msa_mode") if tag.hasOption("msa_mode") else "single_sequence"
        self.rank_ = tag.get_option_string("rank") if tag.hasOption("rank") else "auto"
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else ""
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""
        self.delete_dir_ = tag.get_option_bool("delete_dir") if tag.hasOption("delete_dir") else "0"

        self.tracer_info << (
            f"Parsing options:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\tmodels: {self.models_}\n"
            f"\tmsa_mode: {self.msa_mode_}\n"
            f"\trank: {self.rank_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\tcmd_header: {self.cmd_header_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()

        # RMSD metrics
        for child in tag.getTags():
            if child.getName() == "RMSD":
                self.rmsd_metrics.append({
                    "name": child.get_option_string("name"),
                    "reslabel_input": child.get_option_string("reslabel_input"),
                    "reslabel_prediction": child.get_option_string("reslabel_prediction"),
                    "reslabel_superimpose": (
                        child.get_option_string("reslabel_superimpose")
                        if child.hasOption("reslabel_superimpose") else ""
                    ),
                    "atoms": (
                        child.get_option_string("atoms")
                        if child.hasOption("atoms") else "ca"
                    ),
                })
        # Reject an unknown atoms= value here rather than after the
        # prediction has already run.
        for rmsd in self.rmsd_metrics:
            resolve_rmsd_atoms(rmsd["atoms"])
        rmsd_names = [f"{rmsd['name']} (atoms={rmsd['atoms']})" for rmsd in self.rmsd_metrics]
        self.tracer_info << f"RMSD metrics found: {rmsd_names}\n" and self.tracer_info.flush()

    @staticmethod
    def mover_name():
        return "ColabFold"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_boolean
        
        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        # required attributes
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "cmd_header",
            XMLSchemaType(xs_string),
            "Start to the command string, dependent on user device"))

        # optional attributes
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "replace_pose",
            XMLSchemaType(xs_boolean),
            "Whether current pose is replaced with the rank_001 prediction",
            "1"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "fasta",
            XMLSchemaType(xs_string),
            "Path to a fasta to fold instead of the pose sequence. Each record becomes one pose, with chains of one record separated by colons. The first result enters the pose and the rest come back through get_additional_output(); reslabels and RMSD metrics are skipped, and each pose carries its record id as the comment fasta_id",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "models",
            XMLSchemaType(xs_string),
            "Which of the 5 models to run",
            "1,2,3,4,5"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "msa_mode",
            XMLSchemaType(xs_string),
            "Using an a3m file as input overwrites this option",
            "single_sequence"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "rank",
            XMLSchemaType(xs_string),
            "Which metric to rank models by",
            "auto"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "prefix_name",
            XMLSchemaType(xs_string),
            "Prefix to all metrics calculated by the mover",
            "AF2_"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for the ColabFold executable",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Directory where the ColabFold output will be stored. If not provided, a temporary directory will be used. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete the working directory after the run",
            "0"))
        
        # RMSD attributes
        rmsd_attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        rmsd_attrlist.append(
            XMLSchemaAttribute.required_attribute(
                "name",
                XMLSchemaType(xs_string),
                "RMSD metric name"
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.required_attribute(
                "reslabel_input",
                XMLSchemaType(xs_string),
                "Residue label in input over which RMSD is calculated"
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.required_attribute(
                "reslabel_prediction",
                XMLSchemaType(xs_string),
                "Residue label in prediction over which RMSD is calculated"
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.attribute_w_default(
                "reslabel_superimpose",
                XMLSchemaType(xs_string),
                "Residue label to superimpose over, when it should differ from the residues the RMSD is measured over. For example superimpose on a fixed target chain and measure a designed binder, which reports how well the binder is placed and not only how well it folds. Defaults to the measured residues.",
                ""
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.attribute_w_default(
                "atoms",
                XMLSchemaType(xs_string),
                "Atoms used for both the superposition and the RMSD: ca (alpha carbons only), bb (N, CA, C), bb_o (N, CA, C, O), heavy (backbone and sidechains, no hydrogens), all (every atom), sc or sc_heavy (sidechains only). A core::scoring::rmsd_atoms name is also accepted.",
                "ca"
            )
        )

        description =   '''
                        Runs ColabFold to predict protein structures.
                        '''

        subelements = pyrosetta.rosetta.utility.tag.XMLSchemaSimpleSubelementList()
        subelements.add_simple_subelement("RMSD", rmsd_attrlist, "RMSD applied over specified residue label")
        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes_and_repeatable_subelements(
            xsd, cls.mover_name(), description, attrlist, subelements)


@register_mover
class ColabFoldCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = ColabFold()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return ColabFold.mover_name()

    def provide_xml_schema(self, xsd):
        ColabFold.provide_xml_schema(xsd)