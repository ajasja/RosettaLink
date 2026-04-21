# @file movers/RFDiffusion.py
# @brief Rosetta mover to run RFDiffusion

import os

import pyrosetta
from rosettalink.decorators import register_mover

import tempfile
from pathlib import Path


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
            print(f"[RFDiffusion] No work directory specified, using temporary directory: {self.work_dir_}")
        else:
            os.makedirs(self.work_dir_, exist_ok=True)
        os.makedirs(Path(self.work_dir_)/'schedules', exist_ok=True)
        os.makedirs(Path(self.work_dir_)/'output', exist_ok=True)

        rfdiff_cmd_str = f"singularity run --nv \
            -B {self.work_dir_}:/output \
            {self.rfdiffusion_path_} \
            inference.schedule_directory_path=/output/schedules \
            inference.output_prefix=/output/ \
            'contigmap.contigs={self.contig_}' \
            inference.num_designs={self.num_designs_} \
            -cd /output"   # IMPORTANT: Needs to be within container (with leading slash): self.work_dir_ => /output/
            
        print(f"[RFDiffusion] Running command: {rfdiff_cmd_str}")
        self.run_and_log(rfdiff_cmd_str)
        # Get all .pdb files in the output directory and print their names
        output_dir = Path(self.work_dir_)
        pdb_files = list(output_dir.glob('*.pdb'))
        if not pdb_files:
            print(f"[RFDiffusion] No .pdb files found in output directory {output_dir}")
            raise Exception(f"No .pdb files found in output directory {output_dir}")
        print(f"[RFDiffusion] Found .pdb files: {[str(pdb) for pdb in pdb_files]}")
        pose2 = pyrosetta.pose_from_file(str(pdb_files[0])) #TODO: multi-pose
        pose.assign(pose2)
        try:
            print(f"[RFDiffusion] temp_dir: {temp_dir}")
            temp_dir.cleanup()
            print(f"[RFDiffusion] Cleaned up temporary directory {self.work_dir_}")
        except:
            print(f"[RFDiffusion] It probably wasn't temporary {self.work_dir_}")





    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        print(f"[RFDiffusion] Parsing my tag @ RFDiffusion. Self: {self}, tag: {tag}, data: {data}")
        self.contig_ = tag.get_option_string("contig")
        self.num_designs_ = tag.get_option_int("num_designs")
        self.rfdiffusion_path_ = tag.get_option_string("rfdiffusion_path")
        self.extra_args_ = tag.get_option_string("extra_args")
        self.work_dir_ = tag.get_option_string("work_dir")
        self.delete_dir_ = tag.get_option_bool("delete_dir")

        print(f"[RFDiffusion] Parsed options: contig: {self.contig_}, num_designs: {self.num_designs_}, rfdiffusion_path: {self.rfdiffusion_path_}, extra_args: {self.extra_args_}, work_dir: {self.work_dir_}, delete_dir: {self.delete_dir_}")
    
    def run_and_log(self, command):
        """Runs a command using os.system and also logs the command before running using print"""
        stat = os.system(command)
        wife = os.WIFEXITED(stat)
        exitCode = os.waitstatus_to_exitcode(stat)
        print(f"[RFDiffusion] Command exited with status {stat} and WIFEXITED {wife}. Exit code: {exitCode}")
        if exitCode != 0:
            print("[RFDiffusion] There was an error running the command. We consider it fatal to prevent any file loss. Check the logs and contact the developer.")
            dodatek = ""

            raise Exception(f"[RFDiffusion] Command exited with exit code {exitCode}\n\n{dodatek}")



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
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for the RFDiffusion executable, e.g. --some_flag value"))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "work_dir",
            XMLSchemaType(xs_string),
            "Directory where the RFDiffusion output will be stored"))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete the work directory after the run (what is 'after'? After returning the last pose when being multi-pose?)"))

        description = '''
                        Runs RFDiffusion to generate backbone desings.
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

