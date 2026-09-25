# @file movers/RFDiffusion.py
# @brief Rosetta mover to run RFDiffusion

import os
import shutil

import pyrosetta
from rosettalink.decorators import register_mover
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer
from pyrosetta.rosetta.protocols.residue_selectors import StoreResidueSubsetMover
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, FalseResidueSelector


import tempfile
from pathlib import Path
import pickle
import numpy as np


class RFDiffusion(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(self, contig=None, num_designs=None, rfdiffusion_path=None, extra_args=None, work_dir=None, delete_dir=None):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.contig_ = contig
        self.num_designs_ = num_designs
        self.rfdiffusion_path_ = rfdiffusion_path
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir
        # Populated by apply(): every design beyond the first, exposed to the
        # rest of the Rosetta suite (JD2, RosettaScripts, ...) via the
        # standard Mover::get_additional_output() one-to-many mechanism.
        self.additional_poses_ = None

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[RFDiffusion]")

        self.tracer_info << f"Initialized with contig: {self.contig_}, num_designs: {self.num_designs_}, rfdiffusion_path: {self.rfdiffusion_path_}, extra_args: {self.extra_args_}, work_dir: {self.work_dir_}, delete_dir: {self.delete_dir_} \n" and self.tracer_info.flush()

    def clone(self):
        copy = RFDiffusion()
        copy.contig_ = self.contig_
        copy.num_designs_ = self.num_designs_
        copy.rfdiffusion_path_ = self.rfdiffusion_path_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        RFDiffusion.clones_.append(copy)
        return copy

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
        os.makedirs(run_dir/'schedules', exist_ok=True)

        # Save input pose to run_dir, so we can input it into RfDiff
        pyrosetta.dump_pdb(pose, str(run_dir/'input.pdb'))

        rfdiff_cmd_str = f"singularity run --nv \
            -B {run_dir}:/output \
            {self.rfdiffusion_path_} \
            inference.schedule_directory_path=/output/schedules \
            inference.output_prefix=/output/ \
            'contigmap.contigs={self.contig_}' \
            inference.num_designs={self.num_designs_} \
            {self.extra_args_ if self.extra_args_ else ''} \
            -cd /output"   # IMPORTANT: Needs to be within container (with leading slash): self.work_dir_ => /output/
            
        run_and_log(rfdiff_cmd_str, self.tracer_info, self.tracer_error)
        # RFDiffusion writes a .trb next to every design, and the input pose
        # was dumped into this same directory, so key off the .trb rather
        # than picking up input.pdb as if it were a design.
        output_dir = run_dir
        pdb_files = sorted(
            f for f in output_dir.glob('*.pdb') if f.with_suffix('.trb').is_file()
        )  # _0, _1, _10, _2, _3 ...
        if not pdb_files:
            self.tracer_error << f"No .pdb files found in output directory {output_dir} \n" and self.tracer_error.flush()
            raise Exception(f"No .pdb files found in output directory {output_dir}")
        self.tracer_info << f"Found .pdb files: {[str(pdb) for pdb in pdb_files]} \n" and self.tracer_info.flush()

        designed_poses = [self._load_and_label_design(pdb_file) for pdb_file in pdb_files]

        # Primary output: the pose the caller (RosettaScripts/JD2/plain python)
        # already holds a reference to gets the first design, exactly as before.
        pose.assign(designed_poses[0])

        # Every further design is handed off through Mover::get_additional_output(),
        # the standard Rosetta mechanism for one-to-many movers: JD2's job
        # distributor (and therefore rosetta_scripts, RosettaScripts-driven
        # PyRosetta code, MultiplePoseMover, etc.) automatically drains this
        # after apply() and emits one output structure per pose, exactly like
        # any other native multi-output mover. This replaces silently
        # discarding every design past the first.
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

    def _load_and_label_design(self, pdb_file):
        """Load a single RFDiffusion output .pdb and stamp it with the same
        inpaint_seq/inpaint_str pose-cache subsets and pdb_info reslabels that
        the rest of the suite (selectors, downstream movers/filters) expects,
        regardless of whether this design ends up as the primary output pose
        or as one of the additional ones."""
        pose = pyrosetta.pose_from_file(str(pdb_file))

        # Parse the .trb file
        trb_file = pdb_file.with_suffix('.trb')
        if not trb_file.is_file():
            self.tracer_error << f"No .trb file found for {pdb_file} \n" and self.tracer_error.flush()
            raise Exception(f"No .trb file found for {pdb_file}")
        self.tracer_info << f"Found .trb file: {trb_file} \n" and self.tracer_info.flush()

        with open(trb_file, "rb") as f:
            trb_dict = pickle.load(f)
        residues_to_choose_with_selector_inpaint_seq = trb_dict["inpaint_seq"]
        residues_to_choose_with_selector_inpaint_str = trb_dict["inpaint_str"]
        self.tracer_info << f"Residues to choose with selector: inpaint_seq {residues_to_choose_with_selector_inpaint_seq}; inpaint_str {residues_to_choose_with_selector_inpaint_str} \n" and self.tracer_info.flush()
        resnums_inpaint_seq = ",".join(map(str, (np.nonzero(residues_to_choose_with_selector_inpaint_seq)[0] + 1).tolist())) # Rosetta expects 1-based indices
        resnums_inpaint_str = ",".join(map(str, (np.nonzero(residues_to_choose_with_selector_inpaint_str)[0] + 1).tolist())) # Rosetta expects 1-based indices
        self.tracer_debug << f"Residue numbers to choose with selector: inpaint_seq {resnums_inpaint_seq}; inpaint_str {resnums_inpaint_str} \n" and self.tracer_debug.flush()

        # Residue sets the rest of the suite selects on, in pose numbering:
        #   inpaint_seq   backbone and identity taken from the input
        #   inpaint_str   backbone taken from the input, identity may be new
        #   motif         kept residues placed by the contig
        #   fixed_chain   kept residues outside the motif, i.e. whole chains
        #                 carried through untouched
        #   inpainted     kept backbone whose identity was redesigned
        #   designed      built de novo
        #   all           every residue
        inpaint_seq_resnums = {i + 1 for i, kept in enumerate(residues_to_choose_with_selector_inpaint_seq) if kept}
        inpaint_str_resnums = {i + 1 for i, kept in enumerate(residues_to_choose_with_selector_inpaint_str) if kept}
        motif_resnums = {int(index) + 1 for index in trb_dict.get("con_hal_idx0", [])}
        fixed_chain_resnums = inpaint_str_resnums - motif_resnums
        designed_resnums = set(range(1, len(residues_to_choose_with_selector_inpaint_str) + 1)) - inpaint_str_resnums
        inpainted_resnums = inpaint_str_resnums - inpaint_seq_resnums
        all_resnums = set(range(1, pose.total_residue() + 1))

        def resnum_selector(resnums):
            # ResidueIndexSelector cannot parse an empty string, so an empty
            # set becomes a selector that resolves to nothing selected.
            if not resnums:
                return FalseResidueSelector()
            return ResidueIndexSelector(",".join(str(resnum) for resnum in sorted(resnums)))

        labelled = (
            ("inpaint_seq", inpaint_seq_resnums),
            ("inpaint_str", inpaint_str_resnums),
            ("motif", motif_resnums),
            ("fixed_chain", fixed_chain_resnums),
            ("inpainted", inpainted_resnums),
            ("designed", designed_resnums),
            ("all", all_resnums),
        )

        # Subsets go in the pose cache for StoredResidueSubsetSelector; the
        # same sets go on as pdb_info reslabels for
        # ResiduePDBInfoHasLabelSelector. A label matching no residue is left
        # off entirely rather than stamped as empty.
        for label, resnums in labelled:
            StoreResidueSubsetMover(resnum_selector(resnums), label, True).apply(pose)
            if not resnums:
                continue
            for resnum in sorted(resnums):
                pose.pdb_info().add_reslabel(resnum, label)
            self.tracer_info << f"\t{label}: {len(resnums)} residue(s) \n" and self.tracer_info.flush()

        return pose

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one design per call, None once
        # exhausted. Called by JD2/RosettaScripts (and anything else driving
        # this mover through the standard Mover API) right after apply().
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)



    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ RFDiffusion. Self: {self}, tag: {tag}, data: {data} \n" and self.tracer_debug.flush()
        self.contig_ = tag.get_option_string("contig")
        self.num_designs_ = tag.get_option_int("num_designs")
        self.rfdiffusion_path_ = tag.get_option_string("rfdiffusion_path")
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""
        self.delete_dir_ = tag.get_option_bool("delete_dir")

        self.tracer_info << f"Parsed options: contig: {self.contig_}, num_designs: {self.num_designs_}, rfdiffusion_path: {self.rfdiffusion_path_}, extra_args: {self.extra_args_}, work_dir: {self.work_dir_}, delete_dir: {self.delete_dir_} \n" and self.tracer_info.flush()
    


    @staticmethod
    def mover_name():
        return "RFDiffusion"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_integer, xs_boolean

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "contig",
            XMLSchemaType(xs_string),
            "Contig to design, e.g. [50-100]"))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "num_designs",
            XMLSchemaType(xs_integer),
            "Number of designs to generate"))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "rfdiffusion_path",
            XMLSchemaType(xs_string),
            "Path to the RFDiffusion executable or Docker image"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for the RFDiffusion executable, e.g. diffuser.T=99999",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Directory where the RFDiffusion output will be stored. If attribute not provided, a new tempfile.TemporaryDirectory will be used. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete the work directory after the run (after every design has been read from disk into the primary pose or the additional-output poses)"))

        description = '''
                        Runs RFDiffusion to generate backbone designs.
                      '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd,
            cls.mover_name(),
            description, attrlist)


@register_mover
class RFDiffusionCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = RFDiffusion()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return RFDiffusion.mover_name()

    def provide_xml_schema(self, xsd):
        RFDiffusion.provide_xml_schema(xsd)

