# @file movers/RFDiffusion.py
# @brief Rosetta mover to run RFDiffusion

import os

import pyrosetta
from rosettalink.decorators import register_mover
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer
from pyrosetta.rosetta.protocols.residue_selectors import StoreResidueSubsetMover
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector


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
        if self.work_dir_ is None or self.work_dir_ == "":
            # Create a temporary directory
            temp_dir = tempfile.TemporaryDirectory()
            self.work_dir_ = temp_dir.name
            self.tracer_info << f"No work directory specified, using temporary directory: {self.work_dir_} \n" and self.tracer_info.flush()
        else:
            os.makedirs(self.work_dir_, exist_ok=True)
        os.makedirs(Path(self.work_dir_)/'schedules', exist_ok=True)

        # Save input pose to work_dir, so we can input it into RfDiff
        pyrosetta.dump_pdb(pose, str(Path(self.work_dir_)/'input.pdb'))

        rfdiff_cmd_str = f"singularity run --nv \
            -B {self.work_dir_}:/output \
            {self.rfdiffusion_path_} \
            inference.schedule_directory_path=/output/schedules \
            inference.output_prefix=/output/ \
            'contigmap.contigs={self.contig_}' \
            inference.num_designs={self.num_designs_} \
            {self.extra_args_ if self.extra_args_ else ''} \
            -cd /output"   # IMPORTANT: Needs to be within container (with leading slash): self.work_dir_ => /output/
            
        run_and_log(rfdiff_cmd_str, self.tracer_info, self.tracer_error)
        # Get all .pdb files in the output directory and print their names
        output_dir = Path(self.work_dir_)
        pdb_files = sorted(list(output_dir.glob('*.pdb'))) # _0, _1, _10, _2, _3 ...
        if not pdb_files:
            self.tracer_error << f"No .pdb files found in output directory {output_dir} \n" and self.tracer_error.flush()
            raise Exception(f"No .pdb files found in output directory {output_dir}")
        self.tracer_info << f"Found .pdb files: {[str(pdb) for pdb in pdb_files]} \n" and self.tracer_info.flush()
        for pdb_file in pdb_files:
            pose2 = pyrosetta.pose_from_file(str(pdb_file)) #TODO: multi-pose
            pose.assign(pose2)
            
            # Parse the .trb file
            trb_file = pdb_file.with_suffix('.trb')
            if not trb_file:
                self.tracer_error << f"No .trb file found in output directory {output_dir} \n" and self.tracer_error.flush()
                raise Exception(f"No .trb file found in output directory {output_dir}")
            self.tracer_info << f"Found .trb file: {trb_file} \n" and self.tracer_info.flush()

            with open(trb_file, "rb") as f:
                trb_dict = pickle.load(f)
            residues_to_choose_with_selector_inpaint_seq = trb_dict["inpaint_seq"]
            residues_to_choose_with_selector_inpaint_str = trb_dict["inpaint_str"]
            self.tracer_info << f"Residues to choose with selector: inpaint_seq {residues_to_choose_with_selector_inpaint_seq}; inpaint_str {residues_to_choose_with_selector_inpaint_str} \n" and self.tracer_info.flush()
            resnums_inpaint_seq = ",".join(map(str, (np.nonzero(residues_to_choose_with_selector_inpaint_seq)[0] + 1).tolist())) # Rosetta expects 1-based indices
            resnums_inpaint_str = ",".join(map(str, (np.nonzero(residues_to_choose_with_selector_inpaint_str)[0] + 1).tolist())) # Rosetta expects 1-based indices
            self.tracer_debug << f"Residue numbers to choose with selector: inpaint_seq {resnums_inpaint_seq}; inpaint_str {resnums_inpaint_str} \n" and self.tracer_debug.flush()

            # Store _de novo_ designed residues to pose cache
            inpaint_seq_selector = ResidueIndexSelector(resnums_inpaint_seq)
            inpaint_str_selector = ResidueIndexSelector(resnums_inpaint_str)
            inpaint_seq_srsm = StoreResidueSubsetMover(inpaint_seq_selector, 'inpaint_seq', True)
            inpaint_str_srsm = StoreResidueSubsetMover(inpaint_str_selector, 'inpaint_str', True)
            inpaint_seq_srsm.apply(pose)
            inpaint_str_srsm.apply(pose)

            # Also store inpaint info in pdb labels
            for resnum in map(int, resnums_inpaint_seq.split(",")):
                pose.pdb_info().add_reslabel(resnum, "inpaint_seq")
            for resnum in map(int, resnums_inpaint_str.split(",")):
                pose.pdb_info().add_reslabel(resnum, "inpaint_str")
            
            break

        try:
            self.tracer_debug << f"temp_dir: {temp_dir} \n" and self.tracer_debug.flush()
            temp_dir.cleanup()
            self.tracer_debug << f"Cleaned up temporary directory {self.work_dir_} \n" and self.tracer_debug.flush()
        except:
            self.tracer_debug << f"It probably wasn't temporary {self.work_dir_} \n" and self.tracer_debug.flush()





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
            "Whether to delete the work directory after the run (what is 'after'? After returning the last pose when being multi-pose?)"))

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

