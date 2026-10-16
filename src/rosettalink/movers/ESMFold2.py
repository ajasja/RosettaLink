# @file movers/ESMFold2.py
# @brief Rosetta mover to run ESMFold2
#
# Runs rosettalink/ESM_CLI.py in a separate process rather than importing esm,
# since esm needs python 3.12 or newer. run_command is the whole command
# prefix that reaches ESM_CLI.py:
#
#   run_command="conda run --no-capture-output -p /home/zznidar/.conda/envs/ZZ_ESMFold2 python /path/to/ESM_CLI.py"
#   run_command="singularity run --nv /path/to/esm.sif python /path/to/ESM_CLI.py"
#
# When the attribute is not set the value comes from the ESMFold2 section of
# rosettalink.config.yaml. The mover appends --fasta_file and
# --output_prefix; anything else ESM_CLI.py takes (--model, --device,
# --num_loops, --weights, ...) goes in extra_args.
#
# Otherwise this behaves as the ColabFold mover: each chain of the pose
# becomes its own colon separated entry of the fasta record, every pose of
# the step before is folded in one invocation, and the RMSD and metric
# subtags, subset_selector and fasta work the same way.
#
# Confidence values are reported on the scale ESMFold2 gives them, so pLDDT
# is on 0-1 rather than the 0-100 of AF2.

import json
from pathlib import Path

import pyrosetta

from rosettalink.decorators import register_mover
from rosettalink.utils import get_run_command
from rosettalink.utils import label_prediction
from rosettalink.utils import parse_fasta_records
from rosettalink.utils import parse_prediction_metrics
from rosettalink.utils import pose_records
from rosettalink.utils import run_and_log
from rosettalink.utils import selector_option
from rosettalink.utils import setup_tracer
from rosettalink.utils import work_dir
import rosettalink.movers.BaseLinkMover as BaseLinkMover

# ESM_CLI.py writes <prefix><record index>.cif and <prefix><record index>_scores.json.
OUTPUT_PREFIX = "prediction_"
SCORE_KEYS = ("plddt", "pae", "ptm", "iptm")


class ESMFold2(BaseLinkMover.BaseLinkMover):
    clones_ = list()

    def __init__(
        self,
        replace_pose=True,
        match_chains=True,
        fasta="",
        subset_selector=None,
        prefix_name="ESMFold2_",
        run_command="",
        extra_args=None,
        work_dir=None,
    ):
        BaseLinkMover.BaseLinkMover.__init__(self)
        self.rmsd_metrics = []
        self.confidence_metrics = []
        self.additional_poses_ = []

        self.replace_pose_ = replace_pose
        self.match_chains_ = match_chains
        self.fasta_ = fasta
        self.subset_selector_ = subset_selector
        self.prefix_name_ = prefix_name
        self.run_command_ = run_command
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[ESMFold2]")

    def clone(self):
        copy = ESMFold2()
        copy.replace_pose_ = self.replace_pose_
        copy.match_chains_ = self.match_chains_
        copy.fasta_ = self.fasta_
        copy.subset_selector_ = self.subset_selector_
        copy.prefix_name_ = self.prefix_name_
        copy.run_command_ = self.run_command_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.rmsd_metrics = self.rmsd_metrics.copy()
        copy.confidence_metrics = self.confidence_metrics.copy()
        ESMFold2.clones_.append(copy)
        return copy

    def apply(self, pose):
        run_command = get_run_command(self.mover_name(), self.run_command_)

        per_record = []
        if self.fasta_:
            records = parse_fasta_records(self.fasta_)
        else:
            records, per_record = pose_records(self.take_input_poses(pose), self.subset_selector_)
        self.tracer_info << f"Folding {len(records)} record(s) in one invocation\n" and self.tracer_info.flush()

        with work_dir(self.work_dir_, self.tracer_debug):
            with open("input.fasta", "w") as f:
                for record_id, chains in records:
                    f.write(f">{record_id}\n{':'.join(chains)}\n")

            run_and_log(
                f"{run_command} --fasta_file input.fasta --output_prefix {OUTPUT_PREFIX} "
                f"{self.extra_args_ if self.extra_args_ else ''}",
                self.tracer_info, self.tracer_error,
            )

            predicted_poses = []
            for index, (record_id, _) in enumerate(records):
                input_pose, _, resnums = per_record[index] if per_record else (None, None, None)
                predicted_poses.append(
                    self._load_and_label_prediction(index, record_id, input_pose, resnums)
                )

            pose.assign(predicted_poses[0])
            self.additional_poses_ = list(predicted_poses[1:])

    def _load_and_label_prediction(self, index, record_id, input_pose, folded_resnums):
        """Load the prediction of the index-th record and label it with its
        reslabels, scores and metrics; see utils.label_prediction."""
        structure = Path(f"{OUTPUT_PREFIX}{index}.cif")
        if not structure.is_file():
            raise RuntimeError(f"No ESMFold2 output {structure} found for record {record_id}")
        self.tracer_info << f"{record_id}: {structure}\n" and self.tracer_info.flush()

        scores = None
        scores_file = Path(f"{OUTPUT_PREFIX}{index}_scores.json")
        if scores_file.is_file():
            scores = json.loads(scores_file.read_text())
        else:
            self.tracer_warning << f"No {scores_file} found for record {record_id}\n" and self.tracer_warning.flush()

        prediction = pyrosetta.pose_from_file(str(structure))
        return label_prediction(
            self, prediction, record_id, scores, SCORE_KEYS, input_pose, folded_resnums
        )

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one additional pose per call,
        # None once exhausted.
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.run_command_ = tag.get_option_string("run_command") if tag.hasOption("run_command") else ""
        self.fasta_ = tag.get_option_string("fasta") if tag.hasOption("fasta") else ""
        self.replace_pose_ = tag.get_option_bool("replace_pose") if tag.hasOption("replace_pose") else True
        self.match_chains_ = tag.get_option_bool("match_chains") if tag.hasOption("match_chains") else True
        self.subset_selector_ = selector_option(tag, "subset_selector", data)
        if self.subset_selector_ is not None and self.replace_pose_:
            self.tracer_warning << (
                "subset_selector with replace_pose=true replaces the pose with the "
                "prediction of the subset alone, dropping every residue it does not "
                "pick. Set replace_pose=false to keep the pose and only score it\n"
            ) and self.tracer_warning.flush()
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else "ESMFold2_"
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else None

        self.tracer_info << (
            f"Parsing options:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\trun_command: {self.run_command_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
        ) and self.tracer_info.flush()

        parse_prediction_metrics(self, tag, data)

    @staticmethod
    def mover_name():
        return "ESMFold2"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_boolean

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "run_command",
            XMLSchemaType(xs_string),
            "Command prefix reaching ESM_CLI.py, e.g. conda run --no-capture-output -n ENVNAME python /path/to/ESM_CLI.py. Taken from the ESMFold2 section of rosettalink.config.yaml when not set here",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "replace_pose",
            XMLSchemaType(xs_boolean),
            "Whether current pose is replaced with the prediction",
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
            "subset_selector",
            XMLSchemaType(xs_string),
            "Name of a residue selector naming the only residues to fold, so a binder can be predicted without its target to see whether it keeps its fold alone. Every metric is measured on that prediction. Use replace_pose=false to keep the pose and only score it",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "prefix_name",
            XMLSchemaType(xs_string),
            "Prefix to all metrics calculated by the mover",
            "ESMFold2_"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for ESM_CLI.py, e.g. --model biohub/ESMFold2 --num_loops 3",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Directory where the ESMFold2 output will be stored. If not provided, a temporary directory will be used. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))

        rmsd_attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        rmsd_attrlist.append(XMLSchemaAttribute.required_attribute(
            "name",
            XMLSchemaType(xs_string),
            "RMSD metric name"))
        rmsd_attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "calculation_selector",
            XMLSchemaType(xs_string),
            "Name of a residue selector naming the residues the RMSD is measured over, on both the prediction and the pose it was folded from. Every residue when not set",
            ""))
        rmsd_attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "alignment_selector",
            XMLSchemaType(xs_string),
            "Name of a residue selector naming the residues the structures are superimposed on, when that should differ from the residues the RMSD is measured over. Defaults to the measured residues",
            ""))
        rmsd_attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "atoms",
            XMLSchemaType(xs_string),
            "Atoms used for both the superposition and the RMSD: ca, bb, bb_o, heavy, all, sc or sc_heavy. A core::scoring::rmsd_atoms name is also accepted.",
            "ca"))

        metric_attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        metric_attrlist.append(XMLSchemaAttribute.required_attribute(
            "name",
            XMLSchemaType(xs_string),
            "Score key this metric is reported under, after prefix_name"))
        metric_attrlist.append(XMLSchemaAttribute.required_attribute(
            "base_metric",
            XMLSchemaType(xs_string),
            "Which per residue confidence value is averaged: pLDDT or pAE. pAE is a residue by residue matrix, so a selection averages the block where both residues are selected"))
        metric_attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "selector",
            XMLSchemaType(xs_string),
            "Name of a residue selector naming the residues the confidence value is averaged over. Every residue when not set",
            ""))

        description = "Runs ESMFold2, through ESM_CLI.py, to predict protein structures."

        subelements = pyrosetta.rosetta.utility.tag.XMLSchemaSimpleSubelementList()
        subelements.add_simple_subelement("RMSD", rmsd_attrlist, "Self consistency RMSD between the prediction and the pose it was folded from")
        subelements.add_simple_subelement("metric", metric_attrlist, "Confidence value averaged over the residues a selector picks")
        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes_and_repeatable_subelements(
            xsd, cls.mover_name(), description, attrlist, subelements)


@register_mover
class ESMFold2Creator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = ESMFold2()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return ESMFold2.mover_name()

    def provide_xml_schema(self, xsd):
        ESMFold2.provide_xml_schema(xsd)
