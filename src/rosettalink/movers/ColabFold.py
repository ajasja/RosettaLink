# @file movers/ColabFold.py
# @brief Rosetta mover to run ColabFold

"""
ColabFold command strings should match those in do_cycling.
    reference: https://github.com/ajasja/prosculpt/blob/5211fe061fe0cf03f79a9912fcb9c0f96fc11875/rfdiff_mpnn_af2_merged.py
All scoring metrics should match those in rename_pdb_create_csv_colabfold.
    reference: https://github.com/ajasja/prosculpt/blob/main/prosculpt.py
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import ReturnResidueSubsetSelector
from pyrosetta.rosetta.core.pose import setPoseExtraScore

from rosettalink.decorators import register_mover
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer


class ColabFold(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(
        self, 
        replace_pose=True, 
        models="1,2,3,4,5", 
        msa_mode="single_sequence", 
        rank="auto", 
        prefix_name="AF2_",
        cmd_header=None,
        colabfold_path=None, 
        extra_args=None, 
        work_dir=None, 
        delete_dir=False
    ):        
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)

        self.replace_pose_ = replace_pose
        self.models_ = models
        self.msa_mode_ = msa_mode
        self.rank_ = rank
        self.prefix_name_ = prefix_name
        self.cmd_header_ = cmd_header
        self.colabfold_path_ = colabfold_path
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[ColabFold]")

        self.tracer_info << (
            f"Initializing ColabFold:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\tmodels: {self.models_}\n"
            f"\tmsa_mode: {self.msa_mode_}\n"
            f"\trank: {self.rank_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\tcmd_header: {self.cmd_header_}\n"
            f"\tcolabfold_path: {self.colabfold_path_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()
        
    def clone(self):
        copy = ColabFold()
        copy.replace_pose_ = self.replace_pose_
        copy.models_ = self.models_
        copy.msa_mode_ = self.msa_mode_
        copy.rank_ = self.rank_
        copy.prefix_name_ = self.prefix_name_
        copy.cmd_header_ = self.cmd_header_
        copy.colabfold_path_ = self.colabfold_path_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        ColabFold.clones_.append(copy)
        return copy
    
    def apply(self, pose):
        # --- working directory --- #
        if self.work_dir_ is None or self.work_dir_ == "":
            temp_dir = tempfile.TemporaryDirectory()
            self.work_dir_ = temp_dir.name
            self.tracer_info << f"No working directory specified, using temporary directory: {self.work_dir_}\n" and self.tracer_info.flush()
        else:
            temp_dir = None
            os.makedirs(self.work_dir_, exist_ok=True)
        work_dir = Path(self.work_dir_)

        # --- INPUT FASTA --- #
        sequence = pose.sequence()
        fasta_path = work_dir / "input.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">input\n{sequence}\n")
        self.tracer_info << f"Wrote pose sequence to {fasta_path}\n" and self.tracer_info.flush()

        # --- RUN COLABFOLD --- #
        colabfold_cmd_str = (
            f"{self.cmd_header_} {self.colabfold_path_} "
            f"--model-order {self.models_} "
            f"--msa-mode {self.msa_mode_} "
            f"--rank {self.rank_} "
            f"{self.extra_args_ if self.extra_args_ else ''} "
            f"{fasta_path} "
            f"{work_dir}"
        )

        self.tracer_info << f"Running ColabFold: {colabfold_cmd_str}\n" and self.tracer_info.flush()
        run_and_log(colabfold_cmd_str, self.tracer_info, self.tracer_error)

        # --- OUTPUT PDB --- #
        pdb_files = sorted(work_dir.glob("*.pdb"))
        if not pdb_files:
            self.tracer_error << f"No PDB files found in output directory: {work_dir}\n" and self.tracer_error.flush()
            raise RuntimeError(f"No PDB files found in output directory: {work_dir}")
        self.tracer_info << f"PDB files found: {[str(pdb) for pdb in pdb_files]}\n" and self.tracer_info.flush()

        # --- BEST PREDICTION --- #
        best_pdb = next((pdb for pdb in pdb_files if "rank_001" in pdb.name), pdb_files[0])
        self.tracer_info << f"Best ColabFold prediction: {best_pdb}\n" and self.tracer_info.flush()

        input_pose = pose.clone()
        if self.replace_pose_:
            best_pose = pyrosetta.pose_from_file(str(best_pdb))
            pose.assign(best_pose)
            self.tracer_info << f"Replaced pose with best ColabFold prediction: {best_pdb}\n" and self.tracer_info.flush()

        # --- RMSD METRICS --- #
        for rmsd in self.rmsd_metrics:
            input_selector = ReturnResidueSubsetSelector(rmsd["residue_selector_input"])
            prediction_selector = ReturnResidueSubsetSelector(rmsd["residue_selector_prediction"])

            rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
            rmsd_metric.set_residue_selector(prediction_selector)
            rmsd_metric.set_residue_selector_reference(input_selector)
            rmsd_metric.set_comparison_pose(input_pose)  

            rmsd_value = rmsd_metric.calculate(pose)
            rmsd_name = f"{self.prefix_name_}{rmsd['name']}"
            setPoseExtraScore(pose, rmsd_name, rmsd_value)
            self.tracer_info << f"Stored {rmsd_name}: {rmsd_value}\n" and self.tracer_info.flush()

        # --- STORE SCORES --- #
        json_files = sorted(work_dir.glob("*scores*.json"))
        if not json_files:
            self.tracer_warning << f"No JSON files with scores found in output directory: {work_dir}\n" and self.tracer_warning.flush()
        else:
            best_json = next((j for j in json_files if "rank_001" in j.name), json_files[0])
            self.tracer_info << f"Extracting scores from {best_json}\n" and self.tracer_info.flush()

            with open(best_json, "r") as f:
                scores = json.load(f)

                if "plddt" in scores:
                    # plddt_all
                    plddt = float(np.mean(scores["plddt"]))
                    setPoseExtraScore(pose, f"{self.prefix_name_}plddt", plddt)
                    self.tracer_info << f"\t{self.prefix_name_}plddt: {plddt}\n" and self.tracer_info.flush()

                    # plddt_sculpted
                    sculpted = ReturnResidueSubsetSelector("sculpted")
                    residues_to_exclude = set(np.nonzero(sculpted.apply(pose))[0])  
                    plddt_per_residue = [scores["plddt"][i] for i in range(len(scores["plddt"])) if i not in residues_to_exclude]
                    plddt_sculpted = float(np.mean(plddt_per_residue))
                    setPoseExtraScore(pose, f"{self.prefix_name_}plddt_sculpted", plddt_sculpted)
                    self.tracer_info << f"\t{self.prefix_name_}plddt_sculpted: {plddt_sculpted}\n" and self.tracer_info.flush()  

                if "pae" in scores:
                    # pae_all
                    pae = float(np.mean(scores["pae"]))
                    setPoseExtraScore(pose, f"{self.prefix_name_}pae", pae)
                    self.tracer_info << f"\t{self.prefix_name_}pae: {pae}\n" and self.tracer_info.flush()

        # --- CLEANUP --- #
        if temp_dir:
            temp_dir.cleanup()
            self.tracer_info << f"Cleaned up temporary directory: {self.work_dir_}\n" and self.tracer_info.flush()
        elif self.delete_dir_:
            try:
                shutil.rmtree(self.work_dir_)
                self.tracer_info << f"Deleted working directory: {self.work_dir_}\n" and self.tracer_info.flush()
            except:
                self.tracer_error << f"Failed to delete working directory: {self.work_dir_}\n" and self.tracer_error.flush()

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ ColabFold...\n\tself: {self}\n\ttag: {tag}\n\tdata: {data}\n" and self.tracer_debug.flush()

        # required attributes
        self.cmd_header_ = tag.get_option_string("cmd_header")
        self.colabfold_path_ = tag.get_option_string("colabfold_path")

        # optional attributes
        self.replace_pose_ = tag.get_option_bool("replace_pose") if tag.hasOption("replace_pose") else True
        self.models_ = tag.get_option_string("models") if tag.hasOption("models") else "1,2,3,4,5"
        self.msa_mode_ = tag.get_option_string("msa_mode") if tag.hasOption("msa_mode") else "single_sequence"
        self.rank_ = tag.get_option_string("rank") if tag.hasOption("rank") else "auto"
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else ""
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""
        self.delete_dir_ = tag.get_option_bool("delete_dir") if tag.hasOption("delete_dir") else False

        self.tracer_info << (
            f"Parsed options:\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\tmodels: {self.models_}\n"
            f"\tmsa_mode: {self.msa_mode_}\n"
            f"\trank: {self.rank_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\tcmd_header: {self.cmd_header_}\n"
            f"\tcolabfold_path: {self.colabfold_path_}\n"
            f"\textra_args: {self.extra_args_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()

        # RMSD metrics
        self.rmsd_metrics = []
        for child in tag.getTags():
            if child.getName() == "RMSD":
                self.rmsd_metrics.append({
                    "name": child.get_option_string("name"),
                    "residue_selector_input": child.get_option_string("residue_selector_input"),
                    "residue_selector_prediction": child.get_option_string("residue_selector_prediction")
                })
        rmsd_names = [rmsd["name"] for rmsd in self.rmsd_metrics]
        self.tracer_info << f"Parsed RMSD metrics: {rmsd_names}\n" and self.tracer_info.flush()

    @staticmethod
    def mover_name():
        return "ColabFold"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_boolean
        
        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        # required attributes
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "cmd_header",
            XMLSchemaType(xs_string),
            "Start to the command string, dependent on user device"))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "colabfold_path",
            XMLSchemaType(xs_string),
            "Path to the ColabFold executable or container image"))

        # optional attributes
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "replace_pose",
            XMLSchemaType(xs_boolean),
            "Whether current pose is replaced with the rank_001 prediction",
            1))
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
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete the working directory after the run",
            False))

        description =   '''
                        Runs ColabFold to predict protein structures.
                        '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes( 
            xsd, cls.mover_name(), description, attrlist)
        

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