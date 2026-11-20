# @file movers/Boltz2.py
# @brief Rosetta mover to run Boltz2 (structure prediction), as an
#        alternative to the ColabFold mover.
#
# boltz is a plain CLI (`boltz predict ...`), not a singularity container -
# cmd_header is the full command prefix, e.g. "boltz predict" if it is on
# PATH, or "conda run -n <env> boltz predict" to run it from another
# environment. Any installation-specific setup (proxies, module loads, etc.)
# belongs in cmd_header, not in this mover.
#
# use_msa_server toggles auto-generated MSA via the mmseqs2 server
# (--use_msa_server). When false, the mover writes the "msa: empty" sentinel
# into the input YAML to run in single-sequence mode instead.
#
# recycling_steps/sampling_steps are passed through only when explicitly
# set, otherwise boltz own default applies.
#
# Each chain of the pose is written as its own protein entry (ids A, B, C...
# in pose order), so a complex is folded as a complex.
#
# Every pose the step before produced is written as its own entry, and all of
# them are folded in one boltz invocation.
#
# fasta folds the records of a file instead of the pose sequence: one pose
# per record, with chains of a record separated by colons. All records are
# folded in one boltz invocation. Each pose carries its record id as the
# comment fasta_id, and since these structures have no correspondence to the
# input pose, reslabels and RMSD metrics are skipped.
#
# Confidence scores (confidence_score, ptm, iptm, complex_plddt,
# complex_iplddt, complex_pde, complex_ipde) are reported on Boltz native
# 0-1 scale under their own names, not rescaled to match AF2 pLDDT.
#
# A metric subtag reports one per residue confidence value averaged over the
# residues a selector picks: base_metric is plddt, pae or pde. pae and pde are
# residue by residue matrices, so a selection averages the block where both
# residues are selected.
#
# Nested <RMSD> tags compare the prediction against the pose the mover was
# handed, over residues carrying the given labels. The structures are
# superimposed first, over those same residues. A label matching no residue
# is skipped with a warning rather than failing the run.
#
# atoms="ca" (the default) aligns and measures on alpha carbons only; "heavy"
# or "all" include sidechains. See resolve_rmsd_atoms in rosettalink.utils.

import json
import os
from pathlib import Path

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pose import setPoseExtraScore

from rosettalink.decorators import register_mover
from rosettalink.movers import BaseLinkMover
from rosettalink.utils import carry_over
from rosettalink.utils import parse_fasta_records
from rosettalink.utils import resolve_rmsd_atoms
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer
from rosettalink.utils import work_dir

# Per residue confidence arrays boltz writes beside each prediction, which a
# metric subtag averages over a selection.
PER_RESIDUE_QUANTITIES = ("plddt", "pae", "pde")


class Boltz2(BaseLinkMover.BaseLinkMover):
    clones_ = list()

    def __init__(
        self,
        cmd_header=None,
        cache=None,
        fasta="",
        use_msa_server="1",
        diffusion_samples="1",
        recycling_steps=None,
        sampling_steps=None,
        prefix_name="Boltz_",
        replace_pose="1",
        extra_args=None,
        work_dir=None,
    ):
        BaseLinkMover.BaseLinkMover.__init__(self)
        self.rmsd_metrics = []
        self.confidence_metrics = []

        self.cmd_header_ = cmd_header
        self.cache_ = cache
        self.fasta_ = fasta
        self.use_msa_server_ = use_msa_server
        self.diffusion_samples_ = diffusion_samples
        self.recycling_steps_ = recycling_steps
        self.sampling_steps_ = sampling_steps
        self.prefix_name_ = prefix_name
        self.replace_pose_ = replace_pose
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        # Populated by apply(): every design beyond the first (diffusion_samples > 1),
        # exposed via the standard Mover::get_additional_output() mechanism -
        # same one-to-many pattern as RFdiffusion/LigandMPNN.
        self.additional_poses_ = []

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[Boltz2]")

        self.tracer_info << (
            f"Initializing Boltz2:\n"
            f"\tcmd_header: {self.cmd_header_}\n"
            f"\tcache: {self.cache_}\n"
            f"\tuse_msa_server: {self.use_msa_server_}\n"
            f"\tdiffusion_samples: {self.diffusion_samples_}\n"
            f"\trecycling_steps: {self.recycling_steps_}\n"
            f"\tsampling_steps: {self.sampling_steps_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
        ) and self.tracer_info.flush()

    def clone(self):
        copy = Boltz2()
        copy.cmd_header_ = self.cmd_header_
        copy.cache_ = self.cache_
        copy.fasta_ = self.fasta_
        copy.use_msa_server_ = self.use_msa_server_
        copy.diffusion_samples_ = self.diffusion_samples_
        copy.recycling_steps_ = self.recycling_steps_
        copy.sampling_steps_ = self.sampling_steps_
        copy.prefix_name_ = self.prefix_name_
        copy.replace_pose_ = self.replace_pose_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.rmsd_metrics = self.rmsd_metrics.copy()
        copy.confidence_metrics = self.confidence_metrics.copy()
        Boltz2.clones_.append(copy)
        return copy

    def apply(self, pose):
        # Every pose the step before produced, or just the one passed in.
        # Taken first: pulling runs an intervening FOR_EACH_POSE.
        input_poses = self.take_input_poses(pose)

        # --- INPUT YAML --- #
        # One yaml per entry, each with one protein entry per chain, so a
        # complex is folded as a complex rather than as one fused chain.
        # Chain ids are assigned A, B, C... in that order, which keeps the
        # residue numbering of the prediction aligned with the input pose.
        #
        # boltz is pointed at the directory holding the yamls, so every entry
        # is folded in one invocation. With fasta set the entries come from
        # that file; otherwise there is one entry per input pose.
        #
        # record_poses holds the pose each entry was written from: the source
        # of the reslabels carried forward (boltz output pdbs carry no
        # pdb_info at all) and the comparison pose for the RMSD block. A
        # fasta record has no input pose, so its entry is None.
        if self.fasta_:
            records = parse_fasta_records(self.fasta_)
            record_poses = [None] * len(records)
            self.tracer_info << (
                f"Folding {len(records)} record(s) from {self.fasta_}\n"
            ) and self.tracer_info.flush()
        else:
            records = [
                (f"design_{index}", [chain.sequence() for chain in input_pose.split_by_chain()])
                for index, input_pose in enumerate(input_poses)
            ]
            record_poses = list(input_poses)
            self.tracer_info << (
                f"Folding {len(records)} pose(s) in one invocation\n"
            ) and self.tracer_info.flush()

        with work_dir(self.work_dir_, self.tracer_debug) as run_dir:
            # We are already in run_dir at this point, so every path handed to
            # boltz stays relative to it.
            input_path = Path("inputs")
            os.makedirs(input_path, exist_ok=True)

            for entry_name, chain_sequences in records:
                yaml_path = input_path / f"{entry_name}.yaml"
                with open(yaml_path, "w") as f:
                    f.write("version: 1\n")
                    f.write("sequences:\n")
                    for chain_index, sequence in enumerate(chain_sequences):
                        f.write("  - protein:\n")
                        f.write(f"      id: {chr(ord('A') + chain_index)}\n")
                        f.write(f"      sequence: \"{sequence}\"\n")
                        if not self.use_msa_server_:
                            # Single-sequence mode requires this sentinel, not
                            # just omitting the msa field.
                            f.write("      msa: empty\n")
                self.tracer_info << (
                    f"Writing {len(chain_sequences)} chain(s) to {yaml_path}\n"
                ) and self.tracer_info.flush()

            # --- RUN BOLTZ --- #
            boltz_cmd_str = f"{self.cmd_header_} {input_path} --out_dir . --output_format pdb"
            if self.cache_:
                boltz_cmd_str += f" --cache {self.cache_}"
            if self.use_msa_server_:
                boltz_cmd_str += " --use_msa_server"
            if self.diffusion_samples_:
                boltz_cmd_str += f" --diffusion_samples {self.diffusion_samples_}"
            if self.recycling_steps_:
                boltz_cmd_str += f" --recycling_steps {self.recycling_steps_}"
            if self.sampling_steps_:
                boltz_cmd_str += f" --sampling_steps {self.sampling_steps_}"
            if self.extra_args_:
                boltz_cmd_str += f" {self.extra_args_}"

            self.tracer_info << f"Running Boltz2: {boltz_cmd_str}\n" and self.tracer_info.flush()
            run_and_log(boltz_cmd_str, self.tracer_info, self.tracer_error)

            # --- OUTPUT PDBS --- #
            # Glob for **/predictions/... rather than a fixed path, since boltz
            # nests predictions/ under an extra directory of its own choosing.
            # Entries are walked in input order so the poses come back in the
            # order the records were given.
            output_dir = Path(".")
            designed_poses = []
            for (entry_name, _), record_pose in zip(records, record_poses):
                pdb_files = sorted(
                    output_dir.glob(f"**/predictions/{entry_name}/{entry_name}_model_*.pdb")
                )
                if not pdb_files:
                    self.tracer_error << (
                        f"No boltz output pdb files found for {entry_name} under {run_dir}\n"
                    ) and self.tracer_error.flush()
                    raise RuntimeError(
                        f"No boltz output pdb files found for {entry_name} under {run_dir}"
                    )
                self.tracer_info << (
                    f"Boltz output pdb files for {entry_name}: {[str(p) for p in pdb_files]}\n"
                ) and self.tracer_info.flush()
                record_id = entry_name if self.fasta_ else None
                designed_poses.extend(
                    self._load_and_label_design(pdb_file, record_pose, record_id)
                    for pdb_file in pdb_files
                )

            # Primary output: same convention as RFdiffusion/LigandMPNN - the
            # pose the caller already holds a reference to gets the first design.
            pose.assign(designed_poses[0])

            # Every further design (more input poses, or diffusion_samples > 1)
            # is handed off through Mover::get_additional_output(), the standard
            # Rosetta one-to-many mechanism, instead of being silently discarded.
            self.additional_poses_ = list(designed_poses[1:])

    def _attach_confidence(self, pose, pdb_file, record_id=None):
        """Reads the confidence json boltz writes beside pdb_file and
        attaches its scores to the pose under prefix_name."""
        confidence_path = pdb_file.parent / f"confidence_{pdb_file.stem}.json"
        if not confidence_path.is_file():
            self.tracer_warning << f"No confidence json found at {confidence_path}\n" and self.tracer_warning.flush()
            return
        with open(confidence_path, "r") as f:
            confidence = json.load(f)
        label = f"{record_id} " if record_id else ""
        for key in ("confidence_score", "ptm", "iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"):
            if key in confidence:
                score_name = f"{self.prefix_name_}{key}"
                setPoseExtraScore(pose, score_name, float(confidence[key]))
                self.tracer_info << f"\t{label}{score_name}: {confidence[key]}\n" and self.tracer_info.flush()

    def _load_and_label_design(self, pdb_file, input_pose, record_id=None):
        """Load a single boltz-predicted .pdb, carry over reslabels from the
        input pose, attach confidence scores, and compute any configured
        RMSD metrics. If replace_pose is false, the returned pose keeps the
        original coordinates instead of the boltz-predicted ones.

        record_id marks a structure folded from a fasta record rather than
        from the pose. Those have no correspondence to the input pose, so its
        labels are not carried over and no RMSD is computed; the id is
        recorded as the pose comment fasta_id."""
        if record_id is not None:
            pose = pyrosetta.pose_from_file(str(pdb_file))
            pyrosetta.rosetta.core.pose.add_comment(pose, "fasta_id", record_id)
            self._attach_confidence(pose, pdb_file, record_id)
            self._attach_confidence_metrics(pose, pdb_file, record_id)
            if self.rmsd_metrics:
                self.tracer_warning << (
                    "Skipping RMSD metrics: structures folded from a fasta have no "
                    "correspondence to the input pose\n"
                ) and self.tracer_warning.flush()
            return pose

        if self.replace_pose_:
            pose = pyrosetta.pose_from_file(str(pdb_file))
            carry_over(pose, input_pose)
        else:
            pose = input_pose.clone()

        # --- CONFIDENCE SCORES --- #
        self._attach_confidence(pose, pdb_file)

        # --- RMSD METRICS --- #
        self._attach_rmsd_metrics(pose, input_pose)

        # --- CONFIDENCE METRICS --- #
        self._attach_confidence_metrics(pose, pdb_file)

        return pose

    def _attach_rmsd_metrics(self, prediction, input_pose):
        """Computes every configured <RMSD> between the prediction and the
        pose it was folded from.

        calculation_selector picks the residues it is measured over and
        alignment_selector the residues the structures are superimposed on.
        The prediction keeps the numbering of the pose it was folded from, so
        one selector reads the same residues on both."""
        for rmsd in self.rmsd_metrics:
            measured = self.selected_positions(prediction, rmsd["calculation"])
            if not measured:
                self.tracer_warning << (
                    f"Skipping RMSD {rmsd['name']}: calculation_selector picks no "
                    f"residue of the prediction\n"
                ) and self.tracer_warning.flush()
                continue
            measured_selector = self.index_selector(measured)

            superimpose = None
            if rmsd["alignment"] is not None:
                aligned = self.selected_positions(prediction, rmsd["alignment"])
                if not aligned:
                    self.tracer_warning << (
                        f"Skipping RMSD {rmsd['name']}: alignment_selector picks no "
                        f"residue of the prediction\n"
                    ) and self.tracer_warning.flush()
                    continue
                superimpose = self.index_selector(aligned)

            rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
            rmsd_metric.set_residue_selector(measured_selector)
            rmsd_metric.set_residue_selector_reference(measured_selector)
            rmsd_metric.set_comparison_pose(input_pose)
            if superimpose is not None:
                rmsd_metric.set_residue_selector_super(superimpose)
                rmsd_metric.set_residue_selector_super_reference(superimpose)
            # RMSDMetric does not align by default, which for a prediction in
            # its own reference frame measures the frame offset rather than
            # any structural difference.
            rmsd_metric.set_run_superimpose(True)
            # Governs the superposition as well as the measurement. Rosetta
            # defaults to all heavy atoms, i.e. sidechains included.
            rmsd_metric.set_rmsd_type(resolve_rmsd_atoms(rmsd["atoms"]))

            rmsd_value = rmsd_metric.calculate(prediction)
            rmsd_name = f"{self.prefix_name_}{rmsd['name']}"
            setPoseExtraScore(prediction, rmsd_name, rmsd_value)
            self.tracer_info << f"\t{rmsd_name}: {rmsd_value}\n" and self.tracer_info.flush()

    def _attach_confidence_metrics(self, prediction, pdb_file, record_id=None):
        """Computes every configured <metric>: one per residue confidence
        value boltz wrote beside pdb_file, averaged over the residues the
        metric's selector picks."""
        label = f"{record_id} " if record_id else ""
        for metric in self.confidence_metrics:
            values = self._per_residue_values(pdb_file, metric["quantity"])
            if values is None:
                self.tracer_warning << (
                    f"Skipping metric {metric['name']}: boltz wrote no "
                    f"{metric['quantity']} beside {pdb_file.name}\n"
                ) and self.tracer_warning.flush()
                continue

            value = self.averaged_over_selection(prediction, metric["selector"], values)
            if value is None:
                self.tracer_warning << (
                    f"Skipping metric {metric['name']}: its selector picks no residue\n"
                ) and self.tracer_warning.flush()
                continue

            score_name = f"{self.prefix_name_}{metric['name']}"
            setPoseExtraScore(prediction, score_name, value)
            self.tracer_info << f"\t{label}{score_name}: {value}\n" and self.tracer_info.flush()

    @staticmethod
    def _per_residue_values(pdb_file, quantity):
        """The array boltz writes beside a prediction for one quantity: plddt
        one value per residue, pae and pde a residue by residue matrix. None
        when boltz wrote no such file."""
        path = pdb_file.parent / f"{quantity}_{pdb_file.stem}.npz"
        if not path.is_file():
            return None
        with np.load(path) as data:
            return data[quantity] if quantity in data.files else None

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one additional sample per
        # call, None once exhausted.
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ Boltz2...\n\tself: {self}\n\ttag: {tag}\n\tdata: {data}\n" and self.tracer_debug.flush()

        # required attributes
        self.cmd_header_ = tag.get_option_string("cmd_header")

        # optional attributes
        self.cache_ = tag.get_option_string("cache") if tag.hasOption("cache") else ""
        self.fasta_ = tag.get_option_string("fasta") if tag.hasOption("fasta") else ""
        self.use_msa_server_ = tag.get_option_bool("use_msa_server") if tag.hasOption("use_msa_server") else True
        self.diffusion_samples_ = tag.get_option_string("diffusion_samples") if tag.hasOption("diffusion_samples") else "1"
        self.recycling_steps_ = tag.get_option_string("recycling_steps") if tag.hasOption("recycling_steps") else ""
        self.sampling_steps_ = tag.get_option_string("sampling_steps") if tag.hasOption("sampling_steps") else ""
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else "Boltz_"
        self.replace_pose_ = tag.get_option_bool("replace_pose") if tag.hasOption("replace_pose") else True
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""

        self.tracer_info << (
            f"Parsed options:\n"
            f"\tcmd_header: {self.cmd_header_}\n"
            f"\tcache: {self.cache_}\n"
            f"\tuse_msa_server: {self.use_msa_server_}\n"
            f"\tdiffusion_samples: {self.diffusion_samples_}\n"
            f"\trecycling_steps: {self.recycling_steps_}\n"
            f"\tsampling_steps: {self.sampling_steps_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
        ) and self.tracer_info.flush()

        # RMSD and metric subtags, the same syntax as ColabFold
        for child in tag.getTags():
            if child.getName() == "RMSD":
                self.rmsd_metrics.append(self.parse_rmsd_tag(child, data))
            elif child.getName() == "metric":
                self.confidence_metrics.append(
                    self.parse_metric_tag(child, data, PER_RESIDUE_QUANTITIES)
                )
        # Reject an unknown atoms= value here rather than after the
        # prediction has already run.
        for rmsd in self.rmsd_metrics:
            resolve_rmsd_atoms(rmsd["atoms"])
        rmsd_names = [f"{rmsd['name']} (atoms={rmsd['atoms']})" for rmsd in self.rmsd_metrics]
        self.tracer_info << f"RMSD metrics found: {rmsd_names}\n" and self.tracer_info.flush()
        metric_names = [
            f"{metric['name']} ({metric['quantity']})"
            for metric in self.confidence_metrics
        ]
        self.tracer_info << f"Confidence metrics found: {metric_names}\n" and self.tracer_info.flush()

    @staticmethod
    def mover_name():
        return "Boltz2"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_boolean

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        attrlist.append(XMLSchemaAttribute.required_attribute(
            "cmd_header",
            XMLSchemaType(xs_string),
            "Start of the command string. boltz is a plain CLI, not a singularity container, so this is e.g. boltz predict if it is on PATH, or conda run -n ENVNAME boltz predict to run it from another environment"))

        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "cache",
            XMLSchemaType(xs_string),
            "Path to the boltz cache/model-weights directory (--cache). If not provided, the boltz default (~/.boltz) is used",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "fasta",
            XMLSchemaType(xs_string),
            "Path to a fasta to fold instead of the pose sequence. Each record becomes one pose, with chains of one record separated by colons, and all records are folded in one invocation. The first result enters the pose and the rest come back through get_additional_output(); reslabels and RMSD metrics are skipped, and each pose carries its record id as the comment fasta_id",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "use_msa_server",
            XMLSchemaType(xs_boolean),
            "Whether to auto-generate an MSA via the mmseqs2 server (--use_msa_server)",
            "true"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "diffusion_samples",
            XMLSchemaType(xs_string),
            "Number of diffusion samples to generate (--diffusion_samples). Every sample beyond the first is exposed via get_additional_output()",
            "1"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "recycling_steps",
            XMLSchemaType(xs_string),
            "Number of recycling steps (--recycling_steps). Left unset (boltz default applies) unless explicitly provided",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "sampling_steps",
            XMLSchemaType(xs_string),
            "Number of sampling steps (--sampling_steps). Left unset (boltz default applies) unless explicitly provided",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "prefix_name",
            XMLSchemaType(xs_string),
            "Prefix to all metrics calculated by the mover",
            "Boltz_"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "replace_pose",
            XMLSchemaType(xs_boolean),
            "Whether the current pose is replaced with the boltz-predicted structure",
            "true"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for the boltz predict command",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Parent directory under which each apply call gets its own fresh subdirectory (so this mover instance can safely be applied more than once, e.g. from a RosettaScripts MultiplePoseMover). If not provided, a new temporary directory is used per call instead. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))

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

        description = "Runs Boltz2 to predict protein structures, as an alternative to ColabFold."

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
                "Which per residue confidence value is averaged: plddt, pae or pde. pae and pde are residue by residue matrices, so a selection averages the block where both residues are selected"
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
class Boltz2Creator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = Boltz2()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return Boltz2.mover_name()

    def provide_xml_schema(self, xsd):
        Boltz2.provide_xml_schema(xsd)
