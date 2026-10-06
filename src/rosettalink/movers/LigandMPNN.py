# @file movers/LigandMPNN.py
# @brief Rosetta mover to run LigandMPNN
#
# run_command is the whole command prefix that reaches LigandMPNN's run.py,
# so the mover does not care how it is installed:
#
#   run_command="singularity run --nv /path/to/ligandmpnn.sif"
#   run_command="/path/to/python /path/to/LigandMPNN/run.py"
#
# When the attribute is not set the value comes from the LigandMPNN section of
# rosettalink.config.yaml. Paths are then passed as host paths, so a container
# has to be able to see the working directory - add the bind mount to
# run_command when it is somewhere singularity does not mount by default. The
# deprecated ligandmpnn_path keeps taking a singularity image bound to
# /output, as before.
#
# batch="true" designs sequences for every pose published by the previous
# RosettaLink stage in one invocation, through LigandMPNN --pdb_path_multi,
# instead of one invocation per pose.
#
# Every LigandMPNN command line option is exposed as an attribute of the same
# name and passed through only when set, so LigandMPNN own defaults apply
# otherwise. The exceptions are pdb_path and out_folder, which the mover
# sets, and the *_multi options, which the mover writes itself when batching.
#
# fixed_reslabel and redesigned_reslabel take a residue label instead of a
# residue list, and expand to --fixed_residues and --redesigned_residues.
# This is how a network or motif labelled by an earlier mover is held fixed
# without naming residues by hand.
#
# pack_side_chains="1" makes LigandMPNN build sidechains; the mover then
# reads its packed/ output instead of backbones/. Without it the output
# carries backbone atoms only.
#
# Total sequences per input pose = batch_size * number_of_batches. The first
# enters the pose, the rest come back through get_additional_output().
#
# Reslabels on the input pose are carried onto every designed sequence.

import json
import os
import shutil
import tempfile
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector

from rosettalink.decorators import register_mover
from rosettalink.utils import get_run_command
from rosettalink.utils import pose_buffer
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer

# LigandMPNN command line options, passed through as --name value when set.
# model_type, checkpoint_protein_mpnn, batch_size and number_of_batches have
# their own attributes and are not repeated here.
OPTION_HELP = {
    "checkpoint_ligand_mpnn": "Path to the ligand_mpnn weights",
    "checkpoint_per_residue_label_membrane_mpnn": "Path to the per-residue-label membrane weights",
    "checkpoint_global_label_membrane_mpnn": "Path to the global-label membrane weights",
    "checkpoint_soluble_mpnn": "Path to the soluble_mpnn weights",
    "checkpoint_path_sc": "Path to the sidechain packer weights",
    "fixed_residues": "Residues to hold fixed, space separated, e.g. A12 A13 B2. Combined with any residues from fixed_reslabel",
    "redesigned_residues": "Residues to redesign, space separated; everything else is held fixed. Combined with any residues from redesigned_reslabel",
    "chains_to_design": "Chains to redesign, comma separated, e.g. A,B. All chains by default",
    "parse_these_chains_only": "Chains to parse from the input, comma separated",
    "bias_AA": "Global amino acid bias, e.g. A:-1.024,P:2.34",
    "bias_AA_per_residue": "Path to a json of per-residue amino acid biases",
    "omit_AA": "Amino acids to omit everywhere, e.g. ACG",
    "omit_AA_per_residue": "Path to a json of per-residue amino acids to omit",
    "symmetry_residues": "Residue groups tied together, e.g. A12,A13|C2,C3",
    "symmetry_weights": "Weights matching symmetry_residues, e.g. 1.01,1.0|-1.0,2.0",
    "homo_oligomer": "Set to 1 to apply homo-oligomer symmetry automatically",
    "temperature": "Sampling temperature. LigandMPNN default is 0.1",
    "seed": "Random seed. LigandMPNN default is 0",
    "verbose": "Set to 0 to quieten LigandMPNN own output",
    "save_stats": "Set to 1 to write design statistics",
    "file_ending": "String appended to output file names",
    "zero_indexed": "Set to 1 to start output numbering at 0",
    "fasta_seq_separation": "Character separating chains in the output fasta",
    "ligand_mpnn_use_atom_context": "Set to 0 to hide ligand atoms from the model",
    "ligand_mpnn_use_side_chain_context": "Set to 1 to use sidechains of fixed residues as context",
    "ligand_mpnn_cutoff_for_score": "Angstrom cutoff for scoring protein-context atoms",
    "transmembrane_buried": "Buried residues for the membrane models, space separated",
    "transmembrane_interface": "Interface residues for the membrane models, space separated",
    "global_transmembrane_label": "Set to 1 to label the whole input as transmembrane",
    "parse_atoms_with_zero_occupancy": "Set to 1 to keep zero-occupancy atoms",
    "pack_side_chains": "Set to 1 to build sidechains. The mover then reads the packed output",
    "number_of_packs_per_design": "Sidechain packing samples per design. LigandMPNN default is 4",
    "sc_num_denoising_steps": "Denoising steps during sidechain packing",
    "sc_num_samples": "Samples drawn during sidechain packing",
    "repack_everything": "Set to 1 to repack fixed residues as well",
    "pack_with_ligand_context": "Set to 0 to pack without ligand context",
    "force_hetatm": "Set to 1 to write ligand atoms as HETATM",
    "packed_suffix": "Suffix for packed output file names",
}

# Values of pack_side_chains that mean "on".
TRUE_VALUES = ("1", "true", "True", "yes", "on")

DEFAULT_CHECKPOINT = "/app/ligandmpnn/model_params/proteinmpnn_v_48_020.pt"


class LigandMPNN(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(
        self,
        model_type="protein_mpnn",
        checkpoint_protein_mpnn=DEFAULT_CHECKPOINT,
        run_command="",
        ligandmpnn_path=None,
        batch=False,
        batch_size="1",
        number_of_batches="1",
        fixed_reslabel="",
        redesigned_reslabel="",
        extra_args=None,
        work_dir=None,
        delete_dir=None,
        **options
    ):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.model_type_ = model_type
        self.checkpoint_protein_mpnn_ = checkpoint_protein_mpnn
        self.run_command_ = run_command
        self.ligandmpnn_path_ = ligandmpnn_path
        self.batch_ = batch
        # Total sequences designed per input pose = batch_size * number_of_batches.
        self.batch_size_ = batch_size
        self.number_of_batches_ = number_of_batches
        self.fixed_reslabel_ = fixed_reslabel
        self.redesigned_reslabel_ = redesigned_reslabel
        # Pass-through command line options, empty meaning "leave unset".
        self.options_ = {name: options.get(name, "") for name in OPTION_HELP}
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir
        # Populated by apply(): every designed sequence beyond the first,
        # exposed via the standard Mover::get_additional_output() mechanism -
        # same one-to-many pattern as RFDiffusion.
        self.additional_poses_ = []

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[LigandMPNN]")

        self.tracer_info << (
            f"Initializing LigandMPNN:\n"
            f"\tmodel_type: {self.model_type_}\n"
            f"\tcheckpoint_protein_mpnn: {self.checkpoint_protein_mpnn_}\n"
            f"\trun_command: {self.run_command_}\n"
            f"\tligandmpnn_path: {self.ligandmpnn_path_}\n"
            f"\tbatch: {self.batch_}\n"
            f"\tbatch_size: {self.batch_size_}\n"
            f"\tnumber_of_batches: {self.number_of_batches_}\n"
            f"\tfixed_reslabel: {self.fixed_reslabel_}\n"
            f"\tredesigned_reslabel: {self.redesigned_reslabel_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()

    def clone(self):
        copy = LigandMPNN()
        copy.model_type_ = self.model_type_
        copy.checkpoint_protein_mpnn_ = self.checkpoint_protein_mpnn_
        copy.run_command_ = self.run_command_
        copy.ligandmpnn_path_ = self.ligandmpnn_path_
        copy.batch_ = self.batch_
        copy.batch_size_ = self.batch_size_
        copy.number_of_batches_ = self.number_of_batches_
        copy.fixed_reslabel_ = self.fixed_reslabel_
        copy.redesigned_reslabel_ = self.redesigned_reslabel_
        copy.options_ = dict(self.options_)
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        LigandMPNN.clones_.append(copy)
        return copy

    @staticmethod
    def _residues_for_label(pose, label):
        """Residues carrying label, in LigandMPNN chain-and-number form
        ("A12 A13"). Empty string when the label is unset or matches
        nothing."""
        if not label:
            return ""
        selected = ResiduePDBInfoHasLabelSelector(label).apply(pose)
        pdb_info = pose.pdb_info()
        if pdb_info is None:
            return ""
        return " ".join(
            f"{pdb_info.chain(resnum)}{pdb_info.number(resnum)}"
            for resnum in range(1, pose.total_residue() + 1)
            if selected[resnum]
        )

    def resolve_command(self, run_dir):
        """Returns the command prefix reaching LigandMPNN run.py and the
        directory prefix its arguments use.

        With ligandmpnn_path the image is bound to /output and the arguments
        address that; otherwise run_command is used as given and the
        arguments carry host paths."""
        if not self.run_command_ and self.ligandmpnn_path_:
            self.tracer_warning << (
                "ligandmpnn_path is deprecated, use run_command instead\n"
            ) and self.tracer_warning.flush()
            return (
                f"singularity run --nv -B {run_dir}:/output {self.ligandmpnn_path_}",
                "/output",
            )
        return get_run_command(self.mover_name(), self.run_command_), str(run_dir)

    @staticmethod
    def _write_json(run_dir, base, name, contents):
        """Writes one of the LigandMPNN *_multi json files and returns the
        path to it as the command sees it."""
        with open(run_dir / f"{name}.json", "w") as handle:
            json.dump(contents, handle)
        return f"{base}/{name}.json"

    @staticmethod
    def _merge_residues(*residue_lists):
        """Joins residue lists, dropping duplicates and keeping order."""
        merged = []
        for residue_list in residue_lists:
            for residue in (residue_list or "").split():
                if residue not in merged:
                    merged.append(residue)
        return " ".join(merged)

    def apply(self, pose):
        # Fresh, self-cleaning temp directory every call (or a fresh
        # subdirectory of a configured work_dir) - never stored back onto
        # self.work_dir_, so this mover instance can safely be applied more
        # than once (e.g. from a RosettaScripts MultiplePoseMover).
        if self.work_dir_ is None or self.work_dir_ == "":
            temp_dir = tempfile.TemporaryDirectory()
            run_dir = Path(temp_dir.name)
            self.tracer_info << f"No work directory specified, using temporary directory: {run_dir} \n" and self.tracer_info.flush()
        else:
            temp_dir = None
            os.makedirs(self.work_dir_, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(dir=self.work_dir_))

        # Keep a copy of every pose as handed to us, so reslabels set by
        # earlier movers can be carried forward onto each designed sequence
        # below - LigandMPNN output pdbs carry no pdb_info at all.
        input_poses = pose_buffer.consume(pose, self.tracer_warning) if self.batch_ else [pose]
        input_poses = [input_pose.clone() for input_pose in input_poses]
        self.tracer_info << f"Designing sequences for {len(input_poses)} pose(s)\n" and self.tracer_info.flush()

        command_prefix, base = self.resolve_command(run_dir)

        # One input pdb per pose. The stem identifies which outputs belong to
        # which input, since LigandMPNN names its output after it.
        stems = [f"design_{index}" for index in range(len(input_poses))]
        for stem, input_pose in zip(stems, input_poses):
            pyrosetta.dump_pdb(input_pose, str(run_dir / f"{stem}.pdb"))

        # Residue lists from a label are merged with any given literally, per
        # input pose, since a label resolves to different residues on each.
        residue_lists = {}
        for name, reslabel in (
            ("fixed_residues", self.fixed_reslabel_),
            ("redesigned_residues", self.redesigned_reslabel_),
        ):
            residue_lists[name] = [
                self._merge_residues(
                    self.options_.get(name, ""),
                    self._residues_for_label(input_pose, reslabel),
                )
                for input_pose in input_poses
            ]

        ligmpnn_cmd_str = (
            f"{command_prefix}"
            f" --model_type {self.model_type_}"
            f" --out_folder {base}"
            f" --checkpoint_protein_mpnn {self.checkpoint_protein_mpnn_}"
            f" --batch_size {self.batch_size_}"
            f" --number_of_batches {self.number_of_batches_}"
        )

        if len(input_poses) > 1:
            # --pdb_path_multi takes a json of input pdbs, so every pose is
            # designed in one invocation. The per-pose residue lists have to
            # go through their *_multi counterparts, keyed by the same paths.
            pdb_paths = {f"{base}/{stem}.pdb": "" for stem in stems}
            ligmpnn_cmd_str += f" --pdb_path_multi {self._write_json(run_dir, base, 'pdb_path_multi', pdb_paths)}"
            for name, values in residue_lists.items():
                if not any(values):
                    continue
                keyed = {
                    f"{base}/{stem}.pdb": value
                    for stem, value in zip(stems, values)
                    if value
                }
                ligmpnn_cmd_str += (
                    f" --{name}_multi {self._write_json(run_dir, base, f'{name}_multi', keyed)}"
                )
        else:
            ligmpnn_cmd_str += f" --pdb_path {base}/{stems[0]}.pdb"
            for name, values in residue_lists.items():
                if values[0]:
                    ligmpnn_cmd_str += f' --{name} "{values[0]}"'

        for name, value in self.options_.items():
            # The residue lists are handled above, in whichever form this run
            # needs.
            if name in residue_lists:
                continue
            if value != "" and value is not None:
                # Quoted: symmetry_residues and the residue lists contain
                # pipes and spaces.
                ligmpnn_cmd_str += f' --{name} "{value}"'
        if self.extra_args_:
            ligmpnn_cmd_str += f" {self.extra_args_}"

        for name, values in residue_lists.items():
            for stem, value in zip(stems, values):
                if value:
                    self.tracer_info << f"{stem} {name}: {value}\n" and self.tracer_info.flush()

        run_and_log(ligmpnn_cmd_str, self.tracer_info, self.tracer_error)

        # backbones/ carries backbone atoms only; packed/ is written instead
        # when LigandMPNN builds sidechains.
        packing = str(self.options_.get("pack_side_chains", "")).strip() in TRUE_VALUES
        output_dir = run_dir / ("packed" if packing else "backbones")

        designed_poses = []
        for stem, input_pose in zip(stems, input_poses):
            # The trailing underscore keeps design_1 from picking up the
            # output of design_10.
            pdb_files = sorted(output_dir.glob(f"{stem}_*.pdb"))  # _0, _1, _10, _2, _3 ...
            if not pdb_files and len(input_poses) == 1:
                # With one input there is nothing to tell apart, so accept
                # whatever LigandMPNN named its output.
                pdb_files = sorted(output_dir.glob("*.pdb"))
            if not pdb_files:
                self.tracer_error << f"No .pdb files found for {stem} in output directory {output_dir} \n" and self.tracer_error.flush()
                raise Exception(f"No .pdb files found for {stem} in output directory {output_dir}")
            self.tracer_info << f"Found .pdb files for {stem}: {[str(pdb) for pdb in pdb_files]} \n" and self.tracer_info.flush()
            designed_poses.extend(
                self._load_and_label_design(pdb_file, input_pose) for pdb_file in pdb_files
            )

        # Primary output: the pose the caller already holds a reference to
        # gets the first designed sequence.
        pose.assign(designed_poses[0])

        # Every further sequence is handed off through
        # Mover::get_additional_output(), the standard Rosetta one-to-many
        # mechanism, instead of being silently discarded. While batching they
        # go to the shared buffer instead, for the next RosettaLink mover.
        if self.batch_:
            pose_buffer.publish(self, designed_poses)
            self.additional_poses_ = []
        else:
            self.additional_poses_ = list(designed_poses[1:])

        try:
            if temp_dir:
                temp_dir.cleanup()
                self.tracer_debug << f"Cleaned up temporary directory {run_dir} \n" and self.tracer_debug.flush()
            elif self.delete_dir_:
                shutil.rmtree(run_dir)
                self.tracer_debug << f"Deleted run directory {run_dir} \n" and self.tracer_debug.flush()
        except Exception:
            self.tracer_debug << f"Failed to clean up {run_dir} \n" and self.tracer_debug.flush()

    def _load_and_label_design(self, pdb_file, input_pose):
        """Load one designed sequence and carry over the reslabels of the
        input pose."""
        pose = pyrosetta.pose_from_file(str(pdb_file))

        pdb_info_old = input_pose.pdb_info()
        pdb_info_new = pose.pdb_info()
        if pdb_info_old is not None and pdb_info_new is not None:
            for resnum in range(1, min(input_pose.total_residue(), pose.total_residue()) + 1):
                for reslabel in pdb_info_old.get_reslabels(resnum):
                    pdb_info_new.add_reslabel(resnum, reslabel)

        return pose

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one designed sequence per
        # call, None once exhausted. While batching, the sequences come from
        # the shared buffer and are served only until a downstream mover
        # takes them.
        if self.batch_:
            return pose_buffer.pop_for(self)
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ LigandMPNN. Self: {self}, tag: {tag}, data: {data} \n" and self.tracer_debug.flush()
        self.model_type_ = tag.get_option_string("model_type") if tag.hasOption("model_type") else "protein_mpnn"
        self.checkpoint_protein_mpnn_ = tag.get_option_string("checkpoint_protein_mpnn") if tag.hasOption("checkpoint_protein_mpnn") else DEFAULT_CHECKPOINT
        self.run_command_ = tag.get_option_string("run_command") if tag.hasOption("run_command") else ""
        self.ligandmpnn_path_ = tag.get_option_string("ligandmpnn_path") if tag.hasOption("ligandmpnn_path") else ""
        self.batch_ = tag.get_option_bool("batch") if tag.hasOption("batch") else False
        self.batch_size_ = tag.get_option_string("batch_size") if tag.hasOption("batch_size") else "1"
        self.number_of_batches_ = tag.get_option_string("number_of_batches") if tag.hasOption("number_of_batches") else "1"
        self.fixed_reslabel_ = tag.get_option_string("fixed_reslabel") if tag.hasOption("fixed_reslabel") else ""
        self.redesigned_reslabel_ = tag.get_option_string("redesigned_reslabel") if tag.hasOption("redesigned_reslabel") else ""
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
            f"\tmodel_type: {self.model_type_}\n"
            f"\tcheckpoint_protein_mpnn: {self.checkpoint_protein_mpnn_}\n"
            f"\trun_command: {self.run_command_}\n"
            f"\tligandmpnn_path: {self.ligandmpnn_path_}\n"
            f"\tbatch: {self.batch_}\n"
            f"\tbatch_size: {self.batch_size_}\n"
            f"\tnumber_of_batches: {self.number_of_batches_}\n"
            f"\tfixed_reslabel: {self.fixed_reslabel_}\n"
            f"\tredesigned_reslabel: {self.redesigned_reslabel_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
            f"\tLigandMPNN options: {set_options}\n"
        ) and self.tracer_info.flush()

    @staticmethod
    def mover_name():
        return "LigandMPNN"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_integer, xs_boolean

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "model_type",
            XMLSchemaType(xs_string),
            "Model type",
            "protein_mpnn"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "checkpoint_protein_mpnn",
            XMLSchemaType(xs_string),
            "Model checkpoint",
            DEFAULT_CHECKPOINT))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "run_command",
            XMLSchemaType(xs_string),
            "Command prefix reaching LigandMPNN run.py, whether as a container (singularity run --nv /path/to/ligandmpnn.sif) or a direct call (/path/to/python /path/to/LigandMPNN/run.py). Taken from the LigandMPNN section of rosettalink.config.yaml when not set here. Paths are passed as host paths, so add any bind mount a container needs to this value",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "ligandmpnn_path",
            XMLSchemaType(xs_string),
            "Deprecated. Path to a singularity image, bound to /output, kept so existing protocols keep working. Use run_command instead",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "batch",
            XMLSchemaType(xs_boolean),
            "Whether to design sequences for every pose published by the previous RosettaLink mover in one invocation, through LigandMPNN --pdb_path_multi, instead of one invocation per pose. Leave false when the mover sits inside a FOR_EACH_POSE or MultiplePoseMover, which hands it one pose at a time",
            "false"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "batch_size",
            XMLSchemaType(xs_integer),
            "Number of sequences to generate per batch. Total sequences designed = batch_size * number_of_batches",
            "1"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "number_of_batches",
            XMLSchemaType(xs_integer),
            "Number of batches to run. Total sequences designed = batch_size * number_of_batches",
            "1"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "fixed_reslabel",
            XMLSchemaType(xs_string),
            "Residue label whose residues are held fixed, expanded into --fixed_residues. Combined with any residues given in fixed_residues",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "redesigned_reslabel",
            XMLSchemaType(xs_string),
            "Residue label whose residues are redesigned while everything else is held fixed, expanded into --redesigned_residues. Combined with any residues given in redesigned_residues",
            ""))

        for name, help_text in OPTION_HELP.items():
            attrlist.append(XMLSchemaAttribute.attribute_w_default(
                name,
                XMLSchemaType(xs_string),
                f"{help_text}. Passed through to LigandMPNN only when set",
                ""))

        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for the LigandMPNN command, for anything this mover does not expose",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Parent directory under which each apply() call gets its own fresh subdirectory (so this mover instance can safely be applied more than once, e.g. from a RosettaScripts MultiplePoseMover). If attribute not provided, a new tempfile.TemporaryDirectory is used per call instead. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete this call run subdirectory after every designed sequence has been read from disk into the primary pose or the additional-output poses. Ignored (always cleaned up) when work_dir is unset, since a temp directory is used instead.",
            "false"))

        description = '''
                        Runs LigandMPNN to design sequences onto the pose backbone.
                      '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd,
            cls.mover_name(),
            description, attrlist)


@register_mover
class LigandMPNNCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = LigandMPNN()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return LigandMPNN.mover_name()

    def provide_xml_schema(self, xsd):
        LigandMPNN.provide_xml_schema(xsd)
