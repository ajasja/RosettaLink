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
# Confidence scores (confidence_score, ptm, iptm, complex_plddt,
# complex_iplddt, complex_pde, complex_ipde) are reported on Boltz native
# 0-1 scale under their own names, not rescaled to match AF2 pLDDT.
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
import shutil
import tempfile
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector
from pyrosetta.rosetta.core.pose import setPoseExtraScore

from rosettalink.decorators import register_mover
from rosettalink.utils import resolve_rmsd_atoms
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer


class Boltz2(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(
        self,
        cmd_header=None,
        cache=None,
        use_msa_server="1",
        diffusion_samples="1",
        recycling_steps=None,
        sampling_steps=None,
        prefix_name="Boltz_",
        replace_pose="1",
        extra_args=None,
        work_dir=None,
        delete_dir=None,
    ):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.rmsd_metrics = []

        self.cmd_header_ = cmd_header
        self.cache_ = cache
        self.use_msa_server_ = use_msa_server
        self.diffusion_samples_ = diffusion_samples
        self.recycling_steps_ = recycling_steps
        self.sampling_steps_ = sampling_steps
        self.prefix_name_ = prefix_name
        self.replace_pose_ = replace_pose
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir
        # Populated by apply(): every design beyond the first (diffusion_samples > 1),
        # exposed via the standard Mover::get_additional_output() mechanism -
        # same one-to-many pattern as RFDiffusion/LigandMPNN.
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
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()

    def clone(self):
        copy = Boltz2()
        copy.cmd_header_ = self.cmd_header_
        copy.cache_ = self.cache_
        copy.use_msa_server_ = self.use_msa_server_
        copy.diffusion_samples_ = self.diffusion_samples_
        copy.recycling_steps_ = self.recycling_steps_
        copy.sampling_steps_ = self.sampling_steps_
        copy.prefix_name_ = self.prefix_name_
        copy.replace_pose_ = self.replace_pose_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        copy.rmsd_metrics = self.rmsd_metrics.copy()
        Boltz2.clones_.append(copy)
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
        # carry forward (boltz output pdbs carry no pdb_info at all) and the
        # comparison pose for the RMSD block below.
        input_pose = pose.clone()

        # --- INPUT YAML --- #
        # A fixed entry name (rather than deriving one from the pose) keeps
        # the output path fully predictable: out_dir/.../predictions/input/...
        entry_name = "input"
        yaml_path = run_dir / f"{entry_name}.yaml"
        # One protein entry per chain, in pose order, so a complex is folded
        # as a complex rather than as one fused chain. Chain ids are assigned
        # A, B, C... in that same order, which keeps the residue numbering of
        # the prediction aligned with the input pose.
        chain_sequences = [chain.sequence() for chain in pose.split_by_chain()]
        with open(yaml_path, "w") as f:
            f.write("version: 1\n")
            f.write("sequences:\n")
            for chain_index, sequence in enumerate(chain_sequences):
                f.write("  - protein:\n")
                f.write(f"      id: {chr(ord('A') + chain_index)}\n")
                f.write(f"      sequence: \"{sequence}\"\n")
                if not self.use_msa_server_:
                    # Single-sequence mode requires this sentinel, not just
                    # omitting the msa field.
                    f.write("      msa: empty\n")
        self.tracer_info << f"Writing {len(chain_sequences)} chain(s) to {yaml_path}\n" and self.tracer_info.flush()

        # --- RUN BOLTZ --- #
        boltz_cmd_str = f"{self.cmd_header_} {yaml_path} --out_dir {run_dir} --output_format pdb"
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
        pdb_files = sorted(run_dir.glob(f"**/predictions/{entry_name}/{entry_name}_model_*.pdb"))
        if not pdb_files:
            self.tracer_error << f"No boltz output pdb files found under {run_dir}\n" and self.tracer_error.flush()
            raise RuntimeError(f"No boltz output pdb files found under {run_dir}")
        self.tracer_info << f"Boltz output pdb files found: {[str(p) for p in pdb_files]}\n" and self.tracer_info.flush()

        designed_poses = [self._load_and_label_design(pdb_file, input_pose) for pdb_file in pdb_files]

        # Primary output: same convention as RFDiffusion/LigandMPNN - the
        # pose the caller already holds a reference to gets the first design.
        pose.assign(designed_poses[0])

        # Every further sample (diffusion_samples > 1) is handed off through
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

    def _load_and_label_design(self, pdb_file, input_pose):
        """Load a single boltz-predicted .pdb, carry over reslabels from the
        input pose, attach confidence scores, and compute any configured
        RMSD metrics. If replace_pose is false, the returned pose keeps the
        original coordinates instead of the boltz-predicted ones."""
        if self.replace_pose_:
            pose = pyrosetta.pose_from_file(str(pdb_file))
            pdb_info_old = input_pose.pdb_info()
            pdb_info_new = pose.pdb_info()
            if pdb_info_old is not None and pdb_info_new is not None:
                for resnum in range(1, min(input_pose.total_residue(), pose.total_residue()) + 1):
                    for reslabel in pdb_info_old.get_reslabels(resnum):
                        pdb_info_new.add_reslabel(resnum, reslabel)
        else:
            pose = input_pose.clone()

        # --- CONFIDENCE SCORES --- #
        confidence_path = pdb_file.parent / f"confidence_{pdb_file.stem}.json"
        if confidence_path.is_file():
            with open(confidence_path, "r") as f:
                confidence = json.load(f)
            for key in ("confidence_score", "ptm", "iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"):
                if key in confidence:
                    score_name = f"{self.prefix_name_}{key}"
                    setPoseExtraScore(pose, score_name, float(confidence[key]))
                    self.tracer_info << f"\t{score_name}: {confidence[key]}\n" and self.tracer_info.flush()
        else:
            self.tracer_warning << f"No confidence json found at {confidence_path}\n" and self.tracer_warning.flush()

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

        return pose

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
        self.use_msa_server_ = tag.get_option_bool("use_msa_server") if tag.hasOption("use_msa_server") else True
        self.diffusion_samples_ = tag.get_option_string("diffusion_samples") if tag.hasOption("diffusion_samples") else "1"
        self.recycling_steps_ = tag.get_option_string("recycling_steps") if tag.hasOption("recycling_steps") else ""
        self.sampling_steps_ = tag.get_option_string("sampling_steps") if tag.hasOption("sampling_steps") else ""
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else "Boltz_"
        self.replace_pose_ = tag.get_option_bool("replace_pose") if tag.hasOption("replace_pose") else True
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""
        self.delete_dir_ = tag.get_option_bool("delete_dir") if tag.hasOption("delete_dir") else False

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
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()

        # RMSD metrics (same nested tag as ColabFold)
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
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete this call run subdirectory after every sample has been read from disk into the primary pose or the additional-output poses",
            "false"))

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

        description = "Runs Boltz2 to predict protein structures, as an alternative to ColabFold."

        subelements = pyrosetta.rosetta.utility.tag.XMLSchemaSimpleSubelementList()
        subelements.add_simple_subelement("RMSD", rmsd_attrlist, "RMSD applied over specified residue label")
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
