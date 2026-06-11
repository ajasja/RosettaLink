# @file movers/LigandMPNN.py
# @brief Rosetta mover to run LigandMPNN

import os

import pyrosetta
from rosettalink.decorators import register_mover
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer
from pyrosetta.rosetta.protocols.residue_selectors import StoreResidueSubsetMover
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.protocols.pose_creation import PoseFromSequenceMover
from pyrosetta.rosetta.protocols.simple_moves import SimpleThreadingMover


import tempfile
from pathlib import Path
import pickle
import numpy as np
from Bio import SeqIO



class LigandMPNN(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(self, model_type="protein_mpnn", checkpoint_protein_mpnn="/app/ligandmpnn/model_params/proteinmpnn_v_48_020.pt", ligandmpnn_path=None, extra_args=None, work_dir=None, delete_dir=None):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.model_type_ = model_type
        self.checkpoint_protein_mpnn_ = checkpoint_protein_mpnn
        self.ligandmpnn_path_ = ligandmpnn_path
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[LigandMPNN]")

        self.tracer_info << f"Initialized with model_type: {self.model_type_}, checkpoint_protein_mpnn: {self.checkpoint_protein_mpnn_}, ligandmpnn_path: {self.ligandmpnn_path_}, extra_args: {self.extra_args_}, work_dir: {self.work_dir_}, delete_dir: {self.delete_dir_} \n" and self.tracer_info.flush()

    def clone(self):
        copy = LigandMPNN()
        copy.model_type_ = self.model_type_
        copy.checkpoint_protein_mpnn_ = self.checkpoint_protein_mpnn_
        copy.ligandmpnn_path_ = self.ligandmpnn_path_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        LigandMPNN.clones_.append(copy)
        return copy

    def apply(self, pose):
        if self.work_dir_ is None or self.work_dir_ == "":
            # Create a temporary directory
            temp_dir = tempfile.TemporaryDirectory()
            self.work_dir_ = temp_dir.name
            self.tracer_info << f"No work directory specified, using temporary directory: {self.work_dir_} \n" and self.tracer_info.flush()
        else:
            os.makedirs(self.work_dir_, exist_ok=True)

        # Save input pose to work_dir, so we can input it into RfDiff
        pyrosetta.dump_pdb(pose, str(Path(self.work_dir_)/'input.pdb'))

        ligmpnn_cmd_str = f"singularity run --nv \
            -B {self.work_dir_}:/output \
            {self.ligandmpnn_path_} \
            --model_type {self.model_type_} \
            --pdb_path /output/input.pdb \
            --out_folder /output \
            --checkpoint_protein_mpnn {self.checkpoint_protein_mpnn_} \
            {self.extra_args_ if self.extra_args_ else ''}"
            
        run_and_log(ligmpnn_cmd_str, self.tracer_info, self.tracer_error)


        """ 
        # Using SimpleThreadingMover to add sidechains
        output_dir = Path(self.work_dir_) / "seqs"
        fa_files = sorted(list(output_dir.glob('*.fa'))) # _0, _1, _10, _2, _3 ...
        if not fa_files:
            self.tracer_error << f"No .fa files found in output directory {output_dir} \n" and self.tracer_error.flush()
            raise Exception(f"No .fa files found in output directory {output_dir}")
        self.tracer_info << f"Found .fa files: {[str(fa) for fa in fa_files]} \n" and self.tracer_info.flush()

        for fa_file in fa_files:
            sequences = []
            with open(fa_file, "r") as file_handle:
                sequences = list(SeqIO.parse(file_handle, "fasta"))
            self.tracer_info << f"Parsed sequences from {fa_file}: {sequences} \n" and self.tracer_info.flush()
            sequences.pop(0) # input seq

            seq = str(sequences[0].seq)
            self.tracer_debug << f"Using sequence {seq} to create pose with SimpleThreadingMover \n" and self.tracer_debug.flush()
            seqs = seq.split(":") # Not supported by ThreadingMover; for-loop it manually
            onebasedindex = 1
            for s in seqs:
                m = SimpleThreadingMover(thread_sequence=s, start_position=onebasedindex)
                m.apply(pose)
                onebasedindex += len(s)

            break """
        


        # Just taking LigandMPNN output pdbs (backbone atoms not present).
        output_dir = Path(self.work_dir_) / "backbones"
        pdb_files = sorted(list(output_dir.glob('*.pdb'))) # _0, _1, _10, _2, _3 ...
        if not pdb_files:
            self.tracer_error << f"No .pdb files found in output directory {output_dir} \n" and self.tracer_error.flush()
            raise Exception(f"No .pdb files found in output directory {output_dir}")
        self.tracer_info << f"Found .pdb files: {[str(pdb) for pdb in pdb_files]} \n" and self.tracer_info.flush()


        for pdb_file in pdb_files:
            pose2 = pyrosetta.pose_from_file(str(pdb_file)) #TODO: multi-pose
            pose.assign(pose2)
                        
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
        self.tracer_debug << f"Parsing my tag @ LigandMPNN. Self: {self}, tag: {tag}, data: {data} \n" and self.tracer_debug.flush()
        self.model_type_ = tag.get_option_string("model_type") if tag.hasOption("model_type") else "protein_mpnn"
        self.checkpoint_protein_mpnn_ = tag.get_option_string("checkpoint_protein_mpnn") if tag.hasOption("checkpoint_protein_mpnn") else "/app/ligandmpnn/model_params/proteinmpnn_v_48_020.pt"
        self.ligandmpnn_path_ = tag.get_option_string("ligandmpnn_path")
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""
        self.delete_dir_ = tag.get_option_bool("delete_dir")

        self.tracer_info << f"Parsed options: model_type: {self.model_type_}, checkpoint_protein_mpnn: {self.checkpoint_protein_mpnn_}, ligandmpnn_path: {self.ligandmpnn_path_}, extra_args: {self.extra_args_}, work_dir: {self.work_dir_}, delete_dir: {self.delete_dir_} \n" and self.tracer_info.flush()
    


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
            "/app/ligandmpnn/model_params/proteinmpnn_v_48_020.pt"))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "ligandmpnn_path",
            XMLSchemaType(xs_string),
            "Path to the LigandMPNN executable or Docker image"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for the LigandMPNN executable, e.g. ???",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Directory where the LigandMPNN output will be stored. If attribute not provided, a new tempfile.TemporaryDirectory will be used. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete the work directory after the run (what is 'after'? After returning the last pose when being multi-pose?)"))

        description = '''
                        Runs LigandMPNN to generate backbone designs.
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

