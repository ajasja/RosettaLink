# @file movers/LigandMPNN.py
# @brief Rosetta mover to run LigandMPNN
#
# Every LigandMPNN command line option is exposed as an attribute of the same
# name and passed through only when set, so LigandMPNN own defaults apply
# otherwise. The exceptions are pdb_path and out_folder, which the mover
# sets, and the *_multi options, which take a json listing several input pdbs
# and so do not apply to a mover that is handed one pose per apply().
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

import os
import shutil
import tempfile
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector

from rosettalink.decorators import register_mover
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
        ligandmpnn_path=None,
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
        self.ligandmpnn_path_ = ligandmpnn_path
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
            f"\tligandmpnn_path: {self.ligandmpnn_path_}\n"
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
        copy.ligandmpnn_path_ = self.ligandmpnn_path_
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

        # Keep a copy of the pose as handed to us, so reslabels set by earlier
        # movers can be carried forward onto every designed sequence below -
        # LigandMPNN output pdbs carry no pdb_info at all.
        input_pose = pose.clone()

        pyrosetta.dump_pdb(pose, str(run_dir / 'input.pdb'))

        # Residue lists from a label are merged with any given literally.
        residue_lists = {
            "fixed_residues": self._merge_residues(
                self.options_.get("fixed_residues", ""),
                self._residues_for_label(pose, self.fixed_reslabel_),
            ),
            "redesigned_residues": self._merge_residues(
                self.options_.get("redesigned_residues", ""),
                self._residues_for_label(pose, self.redesigned_reslabel_),
            ),
        }

        ligmpnn_cmd_str = (
            f"singularity run --nv"
            f" -B {run_dir}:/output"
            f" {self.ligandmpnn_path_}"
            f" --model_type {self.model_type_}"
            f" --pdb_path /output/input.pdb"
            f" --out_folder /output"
            f" --checkpoint_protein_mpnn {self.checkpoint_protein_mpnn_}"
            f" --batch_size {self.batch_size_}"
            f" --number_of_batches {self.number_of_batches_}"
        )
        for name, value in self.options_.items():
            value = residue_lists.get(name, value)
            if value != "" and value is not None:
                # Quoted: symmetry_residues and the residue lists contain
                # pipes and spaces.
                ligmpnn_cmd_str += f' --{name} "{value}"'
        if self.extra_args_:
            ligmpnn_cmd_str += f" {self.extra_args_}"

        for name, value in residue_lists.items():
            if value:
                self.tracer_info << f"{name}: {value}\n" and self.tracer_info.flush()

        run_and_log(ligmpnn_cmd_str, self.tracer_info, self.tracer_error)

        # backbones/ carries backbone atoms only; packed/ is written instead
        # when LigandMPNN builds sidechains.
        packing = str(self.options_.get("pack_side_chains", "")).strip() in TRUE_VALUES
        output_dir = run_dir / ("packed" if packing else "backbones")
        pdb_files = sorted(output_dir.glob('*.pdb'))  # _0, _1, _10, _2, _3 ...
        if not pdb_files:
            self.tracer_error << f"No .pdb files found in output directory {output_dir} \n" and self.tracer_error.flush()
            raise Exception(f"No .pdb files found in output directory {output_dir}")
        self.tracer_info << f"Found .pdb files: {[str(pdb) for pdb in pdb_files]} \n" and self.tracer_info.flush()

        designed_poses = [self._load_and_label_design(pdb_file, input_pose) for pdb_file in pdb_files]

        # Primary output: the pose the caller already holds a reference to
        # gets the first designed sequence.
        pose.assign(designed_poses[0])

        # Every further sequence is handed off through
        # Mover::get_additional_output(), the standard Rosetta one-to-many
        # mechanism, instead of being silently discarded.
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
        # call, None once exhausted.
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ LigandMPNN. Self: {self}, tag: {tag}, data: {data} \n" and self.tracer_debug.flush()
        self.model_type_ = tag.get_option_string("model_type") if tag.hasOption("model_type") else "protein_mpnn"
        self.checkpoint_protein_mpnn_ = tag.get_option_string("checkpoint_protein_mpnn") if tag.hasOption("checkpoint_protein_mpnn") else DEFAULT_CHECKPOINT
        self.ligandmpnn_path_ = tag.get_option_string("ligandmpnn_path")
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
            f"\tligandmpnn_path: {self.ligandmpnn_path_}\n"
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
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "ligandmpnn_path",
            XMLSchemaType(xs_string),
            "Path to the LigandMPNN singularity image"))
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
