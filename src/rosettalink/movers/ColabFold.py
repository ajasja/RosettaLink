# @file movers/ColabFold.py
# @brief Rosetta mover to run ColabFold
#
# run_command is the whole command prefix that reaches colabfold_batch, so the
# mover does not care whether it is a container, a conda environment or a
# binary on PATH:
#
#   run_command="colabfold_batch"
#   run_command="singularity run --nv /path/to/colabfold.sif colabfold_batch"
#   run_command="conda run -n colabfold colabfold_batch"
#
# When the attribute is not set the value comes from the ColabFold section of
# rosettalink.config.yaml. The mover appends --model-order, --msa-mode,
# --rank, the input fasta and the output directory.
#
# Each chain of the pose becomes its own entry of the fasta record, colon
# separated, which is how colabfold_batch denotes a complex.
#
# batch="true" folds every pose published by the previous RosettaLink stage in
# one colabfold_batch invocation instead of one per pose. See
# rosettalink.utils.PoseBuffer.
#
# fasta folds the records of a file instead of the pose sequence: one pose per
# record. Each pose carries its record id as the comment fasta_id, and since
# these structures have no correspondence to the input pose, reslabels and
# RMSD metrics are skipped.

"""
ColabFold command strings should match those in do_cycling.
    reference: https://github.com/ajasja/prosculpt/blob/5211fe061fe0cf03f79a9912fcb9c0f96fc11875/rfdiff_mpnn_af2_merged.py
All scoring metrics should match those in rename_pdb_create_csv_colabfold.
    reference: https://github.com/ajasja/prosculpt/blob/main/prosculpt.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector
from pyrosetta.rosetta.core.pose import setPoseExtraScore

from rosettalink.decorators import register_mover
from rosettalink.utils import get_run_command
from rosettalink.utils import parse_fasta_records
from rosettalink.utils import pose_buffer
from rosettalink.utils import pose_chain_sequences
from rosettalink.utils import resolve_rmsd_atoms
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer
from rosettalink.utils import work_dir
import rosettalink.movers.BaseLinkMover as BaseLinkMover


class ColabFold(BaseLinkMover.BaseLinkMover):
    clones_ = list()

    def __init__(
        self,
        replace_pose=True,
        fasta="",
        batch=False,
        models="1,2,3,4,5",
        msa_mode="single_sequence",
        rank="auto",
        prefix_name="AF2_",
        run_command="",
        cmd_header="",
        extra_args=None,
        work_dir=None,
    ):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.rmsd_metrics = []
        # Populated by apply(): one pose per record beyond the first, exposed
        # through get_additional_output(). Unused while batch is set, where
        # the poses are served from the shared buffer instead.
        self.additional_poses_ = []

        self.replace_pose_ = replace_pose
        self.fasta_ = fasta
        self.batch_ = batch
        self.models_ = models
        self.msa_mode_ = msa_mode
        self.rank_ = rank
        self.prefix_name_ = prefix_name
        self.run_command_ = run_command
        self.cmd_header_ = cmd_header
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[ColabFold]")

        self.tracer_info << (
            f"Initializing ColabFold:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\tbatch: {self.batch_}\n"
            f"\tmodels: {self.models_}\n"
            f"\tmsa_mode: {self.msa_mode_}\n"
            f"\trank: {self.rank_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\trun_command: {self.run_command_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
        ) and self.tracer_info.flush()

    def clone(self):
        copy = ColabFold()
        copy.replace_pose_ = self.replace_pose_
        copy.fasta_ = self.fasta_
        copy.batch_ = self.batch_
        copy.models_ = self.models_
        copy.msa_mode_ = self.msa_mode_
        copy.rank_ = self.rank_
        copy.prefix_name_ = self.prefix_name_
        copy.run_command_ = self.run_command_
        copy.cmd_header_ = self.cmd_header_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.rmsd_metrics = self.rmsd_metrics.copy()
        ColabFold.clones_.append(copy)
        return copy

    def resolve_run_command(self):
        """Command prefix reaching colabfold_batch: run_command, else the
        deprecated cmd_header, else the ColabFold section of the
        configuration file."""
        if not self.run_command_ and self.cmd_header_:
            self.tracer_warning << (
                "cmd_header is deprecated, use run_command instead\n"
            ) and self.tracer_warning.flush()
            return self.cmd_header_
        return get_run_command(self.mover_name(), self.run_command_)

    def apply(self, pose):
        run_command = self.resolve_run_command()

        # --- INPUT FASTA --- #
        # One record per structure to fold, with its chains colon separated,
        # which is how colabfold_batch denotes a complex.
        #
        # With fasta set the records come from that file and have no
        # correspondence to any input pose. Otherwise they come from the
        # poses: every pose of the batch, or just the one handed in.
        if self.fasta_:
            records = parse_fasta_records(self.fasta_)
            input_poses = []
            self.tracer_info << (
                f"Folding {len(records)} record(s) from {self.fasta_}\n"
            ) and self.tracer_info.flush()
        else:
            input_poses = (
                pose_buffer.consume(pose, self.tracer_warning) if self.batch_ else [pose]
            )
            records = [
                (f"design_{index}", pose_chain_sequences(input_pose))
                for index, input_pose in enumerate(input_poses)
            ]
            self.tracer_info << (
                f"Folding {len(records)} pose(s) in one invocation\n"
                if self.batch_ else f"Folding 1 pose\n"
            ) and self.tracer_info.flush()

        with work_dir(self.work_dir_, self.tracer_debug) as run_dir:
            # We are already in tmp_dir (run_dir) at this point, so every path
            # handed to colabfold_batch stays relative to it.
            self.tracer_info << f"Current working directory: {os.getcwd()}\n" and self.tracer_info.flush()

            with open("input.fasta", "w") as f:
                for record_id, chains in records:
                    f.write(f">{record_id}\n{':'.join(chains)}\n")
            self.tracer_info << f"Writing {len(records)} record(s) to input.fasta\n" and self.tracer_info.flush()

            # --- RUN COLABFOLD --- #
            colabfold_cmd_str = (
                f"{run_command} "
                f"--model-order {self.models_} "
                f"--msa-mode {self.msa_mode_} "
                f"--rank {self.rank_} "
                f"{self.extra_args_ if self.extra_args_ else ''} "
                f"input.fasta "
                f"."
            )

            run_and_log(colabfold_cmd_str, self.tracer_info, self.tracer_error)

            # --- ONE POSE PER RECORD --- #
            output_dir = Path(".")
            predicted_poses = []
            for index, (record_id, _) in enumerate(records):
                input_pose = input_poses[index] if input_poses else None
                predicted_poses.append(
                    self._load_and_label_prediction(output_dir, record_id, input_pose)
                )

            # Primary output: the pose the caller already holds a reference to
            # gets the first prediction.
            pose.assign(predicted_poses[0])

            if self.batch_:
                pose_buffer.publish(self, predicted_poses)
                self.additional_poses_ = []
            else:
                self.additional_poses_ = list(predicted_poses[1:])

    def _best_prediction_pdb(self, run_dir, record_id):
        """The rank_001 pdb colabfold_batch wrote for one record. The trailing
        underscore keeps design_1 from matching the output of design_10."""
        matches = sorted(run_dir.glob(f"{record_id}_*rank_001*.pdb"))
        if not matches:
            matches = sorted(run_dir.glob(f"{record_id}_*.pdb"))
        if not matches:
            self.tracer_error << (
                f"No ColabFold output pdb found for record {record_id} in {run_dir}\n"
            ) and self.tracer_error.flush()
            raise RuntimeError(
                f"No ColabFold output pdb found for record {record_id} in {run_dir}"
            )
        return matches[0]

    def _load_and_label_prediction(self, run_dir, record_id, input_pose):
        """Load the best prediction of one record, carry over the reslabels of
        its input pose, attach the confidence scores and compute any
        configured RMSD metrics.

        input_pose is None for a structure folded from a fasta record. Those
        have no correspondence to any pose, so no labels are carried and no
        RMSD is computed; the id is recorded as the pose comment fasta_id."""
        best_pdb = self._best_prediction_pdb(run_dir, record_id)
        self.tracer_info << f"{record_id}: {best_pdb}\n" and self.tracer_info.flush()

        if input_pose is None:
            pose = pyrosetta.pose_from_file(str(best_pdb))
            pyrosetta.rosetta.core.pose.add_comment(pose, "fasta_id", record_id)
            self._attach_scores(pose, run_dir, record_id)
            if self.rmsd_metrics:
                self.tracer_warning << (
                    "Skipping RMSD metrics: structures folded from a fasta have no "
                    "correspondence to the input pose\n"
                ) and self.tracer_warning.flush()
            return pose

        if self.replace_pose_:
            pose = pyrosetta.pose_from_file(str(best_pdb))
            pdb_info_old = input_pose.pdb_info()
            pdb_info_new = pose.pdb_info()
            if pdb_info_old is not None and pdb_info_new is not None:
                for resnum in range(1, min(input_pose.total_residue(), pose.total_residue()) + 1):
                    for reslabel in pdb_info_old.get_reslabels(resnum):
                        pdb_info_new.add_reslabel(resnum, reslabel)
        else:
            pose = input_pose.clone()

        self._attach_scores(pose, run_dir, record_id)
        self._attach_rmsd_metrics(pose, input_pose)
        return pose

    def _attach_scores(self, pose, run_dir, record_id):
        """Attaches the mean pLDDT and PAE of one record, read from the
        rank_001 scores json ColabFold writes for it."""
        json_files = sorted(run_dir.glob(f"{record_id}_*scores*rank_001*.json"))
        if not json_files:
            json_files = sorted(run_dir.glob(f"{record_id}_*scores*.json"))
        if not json_files:
            self.tracer_warning << (
                f"No scores json found for record {record_id} in {run_dir}\n"
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

        # Mean pLDDT over the residues labelled sculpted. The label is left
        # over from prosculpt and matches nothing on a pose labelled by the
        # RFdiffusion mover, so the score is skipped rather than reported as
        # the mean of an empty selection.
        if "plddt" in scores:
            pdb_info = pose.pdb_info()
            per_residue = [
                scores["plddt"][resnum - 1]
                for resnum in range(1, min(pose.total_residue(), len(scores["plddt"])) + 1)
                if pdb_info is not None and pdb_info.res_haslabel(resnum, "sculpted")
            ]
            if per_residue:
                value = float(np.mean(per_residue))
                score_name = f"{self.prefix_name_}plddt_sculpted"
                setPoseExtraScore(pose, score_name, value)
                self.tracer_info << f"\t{record_id} {score_name}: {value}\n" and self.tracer_info.flush()

    def _attach_rmsd_metrics(self, pose, input_pose):
        """Computes every configured <RMSD> against the pose this prediction
        was folded from."""
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

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one additional pose per call,
        # None once exhausted. While batching, the poses come from the shared
        # buffer and are served only until a downstream mover takes them.
        if self.batch_:
            return pose_buffer.pop_for(self)
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ ColabFold...\n\tself: {self}\n\ttag: {tag}\n\tdata: {data}\n" and self.tracer_debug.flush()

        self.run_command_ = tag.get_option_string("run_command") if tag.hasOption("run_command") else ""
        self.cmd_header_ = tag.get_option_string("cmd_header") if tag.hasOption("cmd_header") else ""
        self.fasta_ = tag.get_option_string("fasta") if tag.hasOption("fasta") else ""
        self.batch_ = tag.get_option_bool("batch") if tag.hasOption("batch") else False
        self.replace_pose_ = tag.get_option_bool("replace_pose") if tag.hasOption("replace_pose") else True
        self.models_ = tag.get_option_string("models") if tag.hasOption("models") else "1,2,3,4,5"
        self.msa_mode_ = tag.get_option_string("msa_mode") if tag.hasOption("msa_mode") else "single_sequence"
        self.rank_ = tag.get_option_string("rank") if tag.hasOption("rank") else "auto"
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else "AF2_"
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else None

        self.tracer_info << (
            f"Parsing options:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\tbatch: {self.batch_}\n"
            f"\tmodels: {self.models_}\n"
            f"\tmsa_mode: {self.msa_mode_}\n"
            f"\trank: {self.rank_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\trun_command: {self.run_command_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
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

        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "run_command",
            XMLSchemaType(xs_string),
            "Command prefix reaching colabfold_batch, whether it is a binary on PATH (colabfold_batch), a container (singularity run --nv /path/to/colabfold.sif colabfold_batch) or another environment (conda run -n ENVNAME colabfold_batch). Taken from the ColabFold section of rosettalink.config.yaml when not set here",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "cmd_header",
            XMLSchemaType(xs_string),
            "Deprecated name for run_command, kept so existing protocols keep working",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "batch",
            XMLSchemaType(xs_boolean),
            "Whether to fold every pose published by the previous RosettaLink mover in one colabfold_batch invocation, instead of one invocation per pose. Leave false when the mover sits inside a FOR_EACH_POSE or MultiplePoseMover, which hands it one pose at a time",
            "false"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "replace_pose",
            XMLSchemaType(xs_boolean),
            "Whether current pose is replaced with the rank_001 prediction",
            "true"))
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
