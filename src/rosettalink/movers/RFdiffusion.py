# @file movers/RFdiffusion.py
# @brief Rosetta mover to run RFdiffusion
#
# run_command is the whole command prefix that reaches RFdiffusion's
# run_inference.py, so the mover does not care how it is installed:
#
#   run_command="singularity run --nv /path/to/rfdiffusion.sif"
#   run_command="/path/to/python /path/to/RFdiffusion/scripts/run_inference.py"
#
# When the attribute is not set the value comes from the RFdiffusion section
# of rosettalink.config.yaml.
#
# The mover runs inside a fresh temporary directory (see utils.work_dir), so
# every path it passes is relative to it: the input pose is written to
# input.pdb and the designs land in output/.
#
# prefix_name is put in front of every reslabel the mover stamps, so several
# stages can label the same pose without colliding.
#
# batch="true" takes every pose published by the previous RosettaLink stage
# rather than only the one handed in, and publishes every design for the next
# stage. RFdiffusion reads one structure per invocation, so this composes the
# pipeline without reducing the number of invocations; each input gets its own
# subdirectory.

import os

import pyrosetta
from rosettalink.decorators import register_mover
from rosettalink.utils import get_run_command
from rosettalink.utils import pose_buffer
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer
from rosettalink.utils import work_dir
import rosettalink.movers.BaseLinkMover as BaseLinkMover
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, FalseResidueSelector


from pathlib import Path
import pickle
import numpy as np


class RFdiffusion(BaseLinkMover.BaseLinkMover):
    clones_ = list()

    def __init__(self, contig=None, num_designs=None, run_command="", batch=False, extra_args=None, work_dir=None, prefix_name=""):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.contig_ = contig
        self.num_designs_ = num_designs
        self.run_command_ = run_command
        self.batch_ = batch
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.prefix_name_ = prefix_name
        # Populated by apply(): every design beyond the first, exposed to the
        # rest of the Rosetta suite (JD2, RosettaScripts, ...) via the
        # standard Mover::get_additional_output() one-to-many mechanism.
        # Unused while batch is set, where the designs are served from the
        # shared buffer instead.
        self.additional_poses_ = None

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[RFdiffusion]")

        self.tracer_info << f"Initialized with contig: {self.contig_}, num_designs: {self.num_designs_}, run_command: {self.run_command_}, batch: {self.batch_}, extra_args: {self.extra_args_}, work_dir: {self.work_dir_} \n" and self.tracer_info.flush()

    def clone(self):
        copy = RFdiffusion()
        copy.contig_ = self.contig_
        copy.num_designs_ = self.num_designs_
        copy.run_command_ = self.run_command_
        copy.batch_ = self.batch_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.prefix_name_ = self.prefix_name_
        RFdiffusion.clones_.append(copy)
        return copy

    def apply(self, pose):
        run_command = get_run_command(self.mover_name(), self.run_command_)
        input_poses = pose_buffer.consume(pose, self.tracer_warning) if self.batch_ else [pose]

        with work_dir(self.work_dir_, self.tracer_debug) as run_dir:
            # We are already in tmp_dir (run_dir) at this point.
            self.tracer_info << f"Current working directory: {os.getcwd()} \n" and self.tracer_info.flush()
            self.tracer_info << f"Designing from {len(input_poses)} input pose(s) \n" and self.tracer_info.flush()

            designed_poses = []
            for index, input_pose in enumerate(input_poses):
                # One subdirectory per input structure, so the designs of one
                # do not land among the designs of another.
                design_dir = Path(f"input_{index}") if len(input_poses) > 1 else Path(".")
                designed_poses.extend(self._run_one(input_pose, design_dir, run_command))

            # Primary output: the pose the caller (RosettaScripts/JD2/plain python)
            # already holds a reference to gets the first design, exactly as before.
            pose.assign(designed_poses[0])

            # Every further design is handed off through Mover::get_additional_output(),
            # the standard Rosetta mechanism for one-to-many movers: JD2's job
            # distributor (and therefore rosetta_scripts, RosettaScripts-driven
            # PyRosetta code, MultiplePoseMover, etc.) automatically drains this
            # after apply() and emits one output structure per pose, exactly like
            # any other native multi-output mover. While batching they go to the
            # shared buffer instead, for the next RosettaLink mover to pick up.
            if self.batch_:
                pose_buffer.publish(self, designed_poses)
                self.additional_poses_ = []
            else:
                self.additional_poses_ = list(designed_poses[1:])

    def _run_one(self, input_pose, design_dir, run_command):
        """Runs RFdiffusion once over one input structure and returns every
        design it wrote, loaded and labelled.

        Runs from inside design_dir so every path handed to RFdiffusion stays
        relative, whichever directory the batch put this input in."""
        previous_dir = os.getcwd()
        os.makedirs(design_dir, exist_ok=True)
        os.chdir(design_dir)
        try:
            os.makedirs("output", exist_ok=True)
            os.makedirs("output/schedules", exist_ok=True)

            # Save input pose to the run directory, so we can input it into RfDiff
            pyrosetta.dump_pdb(input_pose, str('input.pdb'))

            rfdiff_cmd_str = f"{run_command} \
                inference.schedule_directory_path=output/schedules \
                inference.output_prefix=output/ \
                'contigmap.contigs={self.contig_}' \
                inference.num_designs={self.num_designs_} \
                {self.extra_args_ if self.extra_args_ else ''} \
                -cd output"   # IMPORTANT: Needs to be outside container (without leading slash)

            run_and_log(rfdiff_cmd_str, self.tracer_info, self.tracer_error)
            # RFdiffusion writes a .trb next to every design, and the input pose
            # was dumped into this same directory, so key off the .trb rather
            # than picking up input.pdb as if it were a design.
            output_dir = Path("output")
            pdb_files = sorted(
                f for f in output_dir.glob('*.pdb') if f.with_suffix('.trb').is_file()
            )  # _0, _1, _10, _2, _3 ...
            if not pdb_files:
                self.tracer_error << f"No .pdb files found in output directory {output_dir} \n" and self.tracer_error.flush()
                raise Exception(f"No .pdb files found in output directory {output_dir}")
            self.tracer_info << f"Found .pdb files: {[str(pdb) for pdb in pdb_files]} \n" and self.tracer_info.flush()

            return [self._load_and_label_design(pdb_file) for pdb_file in pdb_files]
        finally:
            os.chdir(previous_dir)

    def _load_and_label_design(self, pdb_file):
        """Load a single RFdiffusion output .pdb and stamp it with the same
        inpaint_seq/inpaint_str pdb_info reslabels that the rest of the suite
        (selectors, downstream movers/filters) expects, regardless of whether
        this design ends up as the primary output pose or as one of the
        additional ones."""
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

        # Residue sets the rest of the suite selects on, in pose numbering.
        # prefix_name is put in front of each of these names.
        #   new_backbone        built de novo
        #   hidden_sidechains   kept backbone whose identity was redesigned
        #   motif               kept residues placed by the contig
        #   fixed               kept residues outside the motif, i.e. whole
        #                       chains carried through untouched
        #   all                 every residue
        #   inpaint_seq         backbone and identity taken from the input
        #   inpaint_str         backbone taken from the input, identity may be new
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
            ("fixed", fixed_chain_resnums),
            ("hidden_sidechains", inpainted_resnums),
            ("new_backbone", designed_resnums),
            ("all", all_resnums),
        )

        # The sets go on as pdb_info reslabels for
        # ResiduePDBInfoHasLabelSelector. A label matching no residue is left
        # off entirely rather than stamped as empty.
        for label, resnums in labelled:
            if not resnums:
                continue
            label = f"{self.prefix_name_}{label}"
            for resnum in sorted(resnums):
                pose.pdb_info().add_reslabel(resnum, label)
            self.tracer_info << f"\t{label}: {len(resnums)} residue(s) \n" and self.tracer_info.flush()

        return pose

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one design per call, None once
        # exhausted. Called by JD2/RosettaScripts (and anything else driving
        # this mover through the standard Mover API) right after apply().
        # While batching, the designs come from the shared buffer and are
        # served only until a downstream mover takes them.
        if self.batch_:
            return pose_buffer.pop_for(self)
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)



    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ RFdiffusion. Self: {self}, tag: {tag}, data: {data} \n" and self.tracer_debug.flush()
        self.contig_ = tag.get_option_string("contig")
        self.num_designs_ = tag.get_option_int("num_designs")
        self.run_command_ = tag.get_option_string("run_command") if tag.hasOption("run_command") else ""
        self.batch_ = tag.get_option_bool("batch") if tag.hasOption("batch") else False
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else None
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else ""

        self.tracer_info << f"Parsed options: contig: {self.contig_}, num_designs: {self.num_designs_}, run_command: {self.run_command_}, batch: {self.batch_}, extra_args: {self.extra_args_}, work_dir: {self.work_dir_} \n" and self.tracer_info.flush()



    @staticmethod
    def mover_name():
        return "RFdiffusion"

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
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "run_command",
            XMLSchemaType(xs_string),
            "Command that runs RFdiffusion run_inference.py, e.g. singularity run --nv /path/to/rfdiffusion.sif, or /path/to/python /path/to/RFdiffusion/scripts/run_inference.py. Taken from the RFdiffusion section of rosettalink.config.yaml when not set here",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "batch",
            XMLSchemaType(xs_boolean),
            "Whether to take every pose published by the previous RosettaLink mover rather than only the one handed in, and publish every design for the next one. Leave false when the mover sits inside a FOR_EACH_POSE or MultiplePoseMover, which hands it one pose at a time",
            "false"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra arguments for the RFdiffusion executable, e.g. diffuser.T=99999. Paths are relative to the run directory, so the dumped input structure is reached with inference.input_pdb=input.pdb",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "prefix_name",
            XMLSchemaType(xs_string),
            "Put in front of every reslabel this mover stamps, so prefix_name=rfd_ gives rfd_new_backbone, rfd_fixed and so on. Empty by default, which leaves the labels unprefixed",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Directory where the RFdiffusion output will be stored. If attribute not provided, a new tempfile.TemporaryDirectory will be used. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))

        description = '''
                        Runs RFdiffusion to generate backbone designs.
                      '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd,
            cls.mover_name(),
            description, attrlist)


@register_mover
class RFdiffusionCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = RFdiffusion()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return RFdiffusion.mover_name()

    def provide_xml_schema(self, xsd):
        RFdiffusion.provide_xml_schema(xsd)
