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
# Every pose of the step before is folded in one colabfold_batch invocation
# instead of one per pose. See movers.BaseLinkMover.
#
# An RMSD subtag measures a self consistency RMSD: calculation names the
# residues it is measured over and alignment the residues it is superimposed
# on, both by reslabel.
#
# A metric subtag reports one confidence value averaged over the residues
# carrying a reslabel. The name says which value, so plddt_motif averages
# pLDDT and pAE_motif averages PAE, and is also the score key after
# prefix_name.
#
# fasta folds the records of a file instead of the pose sequence: one pose per
# record. Each pose carries its record id as the comment fasta_id, and since
# these structures have no correspondence to the input pose, reslabels and
# RMSD metrics are skipped.

import json
import os
import shutil
from pathlib import Path

import pyrosetta

from rosettalink.decorators import register_mover
from rosettalink.utils import get_run_command
from rosettalink.utils import build_partial_a3m
from rosettalink.utils import chain_letter
from rosettalink.utils import label_prediction
from rosettalink.utils import parse_fasta_records
from rosettalink.utils import parse_prediction_metrics
from rosettalink.utils import pose_records
from rosettalink.utils import run_and_log
from rosettalink.utils import selector_option
from rosettalink.utils import setup_tracer
from rosettalink.utils import work_dir
import rosettalink.movers.BaseLinkMover as BaseLinkMover


class ColabFold(BaseLinkMover.BaseLinkMover):
    clones_ = list()

    def __init__(
        self,
        replace_pose=True,
        match_chains=True,
        fasta="",
        models="1,2,3,4,5",
        msa_mode="single_sequence",
        existing_msa_file="",
        existing_msa_selector=None,
        subset_selector=None,
        rank="auto",
        prefix_name="AF2_",
        run_command="",
        extra_args=None,
        work_dir=None,
    ):
        BaseLinkMover.BaseLinkMover.__init__(self)
        self.rmsd_metrics = []
        self.confidence_metrics = []
        # Populated by apply(): one pose per record beyond the first, exposed
        # through get_additional_output(). Unused while the results go
        # the poses are served from the shared buffer instead.
        self.additional_poses_ = []

        self.replace_pose_ = replace_pose
        self.match_chains_ = match_chains
        self.fasta_ = fasta
        self.models_ = models
        self.msa_mode_ = msa_mode
        self.existing_msa_file_ = existing_msa_file
        self.existing_msa_selector_ = existing_msa_selector
        self.subset_selector_ = subset_selector
        self.rank_ = rank
        self.prefix_name_ = prefix_name
        self.run_command_ = run_command
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[ColabFold]")

        self.tracer_info << (
            f"Initializing ColabFold:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
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
        copy.match_chains_ = self.match_chains_
        copy.fasta_ = self.fasta_
        copy.models_ = self.models_
        copy.msa_mode_ = self.msa_mode_
        copy.existing_msa_file_ = self.existing_msa_file_
        copy.existing_msa_selector_ = self.existing_msa_selector_
        copy.subset_selector_ = self.subset_selector_
        copy.rank_ = self.rank_
        copy.prefix_name_ = self.prefix_name_
        copy.run_command_ = self.run_command_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.rmsd_metrics = self.rmsd_metrics.copy()
        copy.confidence_metrics = self.confidence_metrics.copy()
        ColabFold.clones_.append(copy)
        return copy

    def resolve_run_command(self):
        """Command prefix reaching colabfold_batch: run_command, else the
        ColabFold section of the configuration file."""
        return get_run_command(self.mover_name(), self.run_command_)

    def apply(self, pose):
        run_command = self.resolve_run_command()

        # One record per structure to fold, with its chains colon separated,
        # which is how colabfold_batch denotes a complex.
        #
        # With fasta set the records come from that file and have no
        # correspondence to any input pose. Otherwise they come from the
        # poses handed over by the step before, or just the one passed in.
        #
        # Paths named in the protocol are resolved here, before work_dir()
        # changes directory, so they stay relative to where the run started.
        msa_sources = self._resolve_msa_files()
        per_record = []
        if self.fasta_:
            records = parse_fasta_records(self.fasta_)
            input_poses = []
            self.tracer_info << (
                f"Folding {len(records)} record(s) from {self.fasta_}\n"
            ) and self.tracer_info.flush()
        else:
            records, per_record = pose_records(self.take_input_poses(pose), self.subset_selector_)
            self.tracer_info << (
                f"Folding {len(records)} pose(s) in one invocation"
                f"{self._subset_note()}\n"
            ) and self.tracer_info.flush()

        with work_dir(self.work_dir_, self.tracer_debug) as run_dir:
            # We are already in tmp_dir (run_dir) at this point, so every path
            # handed to colabfold_batch stays relative to it.
            self.tracer_info << f"Current working directory: {os.getcwd()}\n" and self.tracer_info.flush()

            if msa_sources and per_record:
                # colabfold_batch takes a directory of a3m files and uses the
                # alignment in each instead of querying the server.
                staged = self._stage_alignments(msa_sources)
                os.makedirs("msas", exist_ok=True)
                for (record_id, _), (input_pose, chains, _) in zip(records, per_record):
                    alignment = self._partial_a3m(input_pose, chains, staged)
                    with open(os.path.join("msas", f"{record_id}.a3m"), "w") as f:
                        f.write(alignment)
                colabfold_input = "msas"
                msa_option = ""
                self.tracer_info << (
                    f"Wrote {len(records)} alignment(s) to msas/\n"
                ) and self.tracer_info.flush()
            else:
                with open("input.fasta", "w") as f:
                    for record_id, chains in records:
                        f.write(f">{record_id}\n{':'.join(chains)}\n")
                colabfold_input = "input.fasta"
                msa_option = f"--msa-mode {self.msa_mode_} "
                self.tracer_info << (
                    f"Writing {len(records)} record(s) to input.fasta\n"
                ) and self.tracer_info.flush()

            # --- RUN COLABFOLD --- #
            colabfold_cmd_str = (
                f"{run_command} "
                f"--model-order {self.models_} "
                f"{msa_option}"
                f"--rank {self.rank_} "
                f"{self.extra_args_ if self.extra_args_ else ''} "
                f"{colabfold_input} "
                f"."
            )

            run_and_log(colabfold_cmd_str, self.tracer_info, self.tracer_error)

            # --- ONE POSE PER RECORD --- #
            output_dir = Path(".")
            predicted_poses = []
            for index, (record_id, _) in enumerate(records):
                input_pose, _, resnums = per_record[index] if per_record else (None, None, None)
                predicted_poses.append(
                    self._load_and_label_prediction(output_dir, record_id, input_pose, resnums)
                )

            # Primary output: the pose the caller already holds a reference to
            # gets the first prediction.
            pose.assign(predicted_poses[0])

            self.additional_poses_ = list(predicted_poses[1:])

    def _subset_note(self):
        return " over the residues subset_selector picks" if self.subset_selector_ else ""

    def _resolve_msa_files(self):
        """The alignment named for each chain, as absolute paths so they
        survive work_dir() changing directory, and None for a chain given no
        alignment. Empty unless msa_mode is partial."""
        if self.msa_mode_ != "partial":
            return []
        sources = []
        for entry in self.existing_msa_file_.split(","):
            entry = entry.strip()
            if not entry:
                sources.append(None)
                continue
            path = Path(entry).absolute()
            if not path.is_file():
                raise RuntimeError(f"existing_msa_file not found: {path}")
            sources.append(path)
        return sources

    def _stage_alignments(self, sources):
        """Copies each supplied alignment into the run directory and returns
        what it holds, so the run carries the alignments it folded against."""
        os.makedirs("input_msas", exist_ok=True)
        staged = []
        for index, source in enumerate(sources):
            if source is None:
                staged.append(None)
                continue
            local = Path("input_msas") / f"chain_{chain_letter(index)}_{source.name}"
            shutil.copyfile(source, local)
            staged.append(local.read_text())
            self.tracer_info << (
                f"Alignment for chain {chain_letter(index)}: {source} -> {local}\n"
            ) and self.tracer_info.flush()
        return staged

    def _partial_a3m(self, pose, chains, alignments):
        """The alignment for one record, pairing each chain with the one
        supplied for it. Chains given none are folded from their sequence
        alone."""
        if len(alignments) != len(chains):
            raise RuntimeError(
                f"existing_msa_file lists {len(alignments)} entry/entries but the pose "
                f"has {len(chains)} chain(s). Give one entry per chain, in chain order, "
                f"leaving an entry empty for a chain with no alignment"
            )

        covered_by_chain = []
        for _, _, resnums in chains:
            if self.existing_msa_selector_ is None:
                covered_by_chain.append(None)
                continue
            picked = self.existing_msa_selector_.apply(pose)
            covered_by_chain.append(
                [position for position, resnum in enumerate(resnums) if picked[resnum]]
            )

        return build_partial_a3m(
            [sequence for _, sequence, _ in chains], alignments, covered_by_chain
        )

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

    def _load_and_label_prediction(self, run_dir, record_id, input_pose, folded_resnums):
        """Load the best prediction of one record and label it with its
        reslabels, scores and metrics; see utils.label_prediction."""
        best_pdb = self._best_prediction_pdb(run_dir, record_id)
        self.tracer_info << f"{record_id}: {best_pdb}\n" and self.tracer_info.flush()

        prediction = pyrosetta.pose_from_file(str(best_pdb))
        return label_prediction(
            self, prediction, record_id, self._scores(run_dir, record_id),
            ("plddt", "pae"), input_pose, folded_resnums,
        )

    def _scores(self, run_dir, record_id):
        """The rank_001 scores json ColabFold writes for one record, or None
        when there is none."""
        json_files = sorted(run_dir.glob(f"{record_id}_*scores*rank_001*.json"))
        if not json_files:
            json_files = sorted(run_dir.glob(f"{record_id}_*scores*.json"))
        if not json_files:
            self.tracer_warning << (
                f"No scores json found for record {record_id} in {run_dir}\n"
            ) and self.tracer_warning.flush()
            return None

        with open(json_files[0], "r") as f:
            return json.load(f)

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one additional pose per call,
        # None once exhausted.
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ ColabFold...\n\tself: {self}\n\ttag: {tag}\n\tdata: {data}\n" and self.tracer_debug.flush()

        self.run_command_ = tag.get_option_string("run_command") if tag.hasOption("run_command") else ""
        self.fasta_ = tag.get_option_string("fasta") if tag.hasOption("fasta") else ""
        self.replace_pose_ = tag.get_option_bool("replace_pose") if tag.hasOption("replace_pose") else True
        self.match_chains_ = tag.get_option_bool("match_chains") if tag.hasOption("match_chains") else True
        self.models_ = tag.get_option_string("models") if tag.hasOption("models") else "1,2,3,4,5"
        self.msa_mode_ = tag.get_option_string("msa_mode") if tag.hasOption("msa_mode") else "single_sequence"
        self.existing_msa_file_ = tag.get_option_string("existing_msa_file") if tag.hasOption("existing_msa_file") else ""
        self.existing_msa_selector_ = selector_option(tag, "existing_msa_selector", data)
        self.subset_selector_ = selector_option(tag, "subset_selector", data)
        if self.msa_mode_ == "partial" and not self.existing_msa_file_:
            raise RuntimeError(
                "msa_mode=partial needs existing_msa_file, one entry per chain in chain order"
            )
        if self.subset_selector_ is not None and self.replace_pose_:
            self.tracer_warning << (
                "subset_selector with replace_pose=true replaces the pose with the "
                "prediction of the subset alone, dropping every residue it does not "
                "pick. Set replace_pose=false to keep the pose and only score it\n"
            ) and self.tracer_warning.flush()
        self.rank_ = tag.get_option_string("rank") if tag.hasOption("rank") else "auto"
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else "AF2_"
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else None

        self.tracer_info << (
            f"Parsing options:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\tmodels: {self.models_}\n"
            f"\tmsa_mode: {self.msa_mode_}\n"
            f"\trank: {self.rank_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\trun_command: {self.run_command_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
        ) and self.tracer_info.flush()

        parse_prediction_metrics(self, tag, data)

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
            "replace_pose",
            XMLSchemaType(xs_boolean),
            "Whether current pose is replaced with the rank_001 prediction",
            "true"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "match_chains",
            XMLSchemaType(xs_boolean),
            "Whether an RMSD pairs the chains of the prediction with the chains of the pose by fit rather than by name. A predictor names the chains of a homo-oligomer in whatever order it likes, which otherwise compares one chain against another and reports a large number for a structure that matches. Chains pair only with chains of the same sequence, so a complex of unlike chains is unaffected. It changes nothing about the pose the mover emits, and is not applied to an RMSD that sets alignment_selector",
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
            "existing_msa_file",
            XMLSchemaType(xs_string),
            "Alignments in a3m format to fold against instead of querying the alignment server, one comma separated entry per chain in chain order, an empty entry for a chain that has none. Needs msa_mode=partial. Paths are relative to where the protocol was started",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "existing_msa_selector",
            XMLSchemaType(xs_string),
            "Name of a residue selector naming the residues the alignments cover. Every other residue is a gap in the assembled alignment. Every residue when not set, which means each alignment covers its whole chain",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "subset_selector",
            XMLSchemaType(xs_string),
            "Name of a residue selector naming the only residues to fold, so a binder can be predicted without its target to see whether it keeps its fold alone. Every metric is measured on that prediction. Use replace_pose=false to keep the pose and only score it",
            ""))
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
            XMLSchemaAttribute.attribute_w_default(
                "calculation_selector",
                XMLSchemaType(xs_string),
                "Name of a residue selector naming the residues the RMSD is measured over, on both the prediction and the pose it was folded from. Every residue when not set",
                ""
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.attribute_w_default(
                "alignment_selector",
                XMLSchemaType(xs_string),
                "Name of a residue selector naming the residues the structures are superimposed on, when that should differ from the residues the RMSD is measured over. For example superimpose on a fixed target chain and measure a designed binder, which reports how well the binder is placed and not only how well it folds. Defaults to the measured residues",
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

        # metric attributes
        metric_attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        metric_attrlist.append(
            XMLSchemaAttribute.required_attribute(
                "name",
                XMLSchemaType(xs_string),
                "Score key this metric is reported under, after prefix_name"
            )
        )
        metric_attrlist.append(
            XMLSchemaAttribute.required_attribute(
                "base_metric",
                XMLSchemaType(xs_string),
                "Which per residue confidence value is averaged: pLDDT or pAE. pAE is a residue by residue matrix, so a selection averages the block where both residues are selected"
            )
        )
        metric_attrlist.append(
            XMLSchemaAttribute.attribute_w_default(
                "selector",
                XMLSchemaType(xs_string),
                "Name of a residue selector naming the residues the confidence value is averaged over. Every residue when not set",
                ""
            )
        )

        subelements = pyrosetta.rosetta.utility.tag.XMLSchemaSimpleSubelementList()
        subelements.add_simple_subelement("RMSD", rmsd_attrlist, "Self consistency RMSD between the prediction and the pose it was folded from")
        subelements.add_simple_subelement("metric", metric_attrlist, "Confidence value averaged over the residues a selector picks")
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
