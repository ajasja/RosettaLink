# @file movers/RFD4Proteina.py
# @brief Rosetta mover to run RFD4-Proteina
#
# run_command is the whole command prefix reaching the program, e.g.
# "singularity run --nv /path/to/rfd4.sif rfd4-generate". Unset, it comes
# from the RFD4Proteina section of rosettalink.config.yaml.
#
# mode picks the entry point: generate designs structures, evaluate scores
# structures that already exist. Evaluating takes the configuration's
# evaluate_run_command, or run_command with rfd4-generate replaced.
#
# The mover runs inside utils.work_dir, so every path it passes is relative
# to it: inputs/, spec.yaml, output/. Both entry points are given mmCIF.
#
# Settings reach the program as config_file, one yaml holding the whole run
# whose top-level keys become hydra key=value arguments, or spec_file, a
# condition spec on its own; config_file's reserved designs key holds a
# condition spec too. Precedence is config_file, then the attributes that are
# set, then extra_args. inputs and out_dir are the mover's own.
#
# prefix_name is put in front of every reslabel and score key it stamps.
# Every pose of the step before is taken as an input structure.

import csv
import os
import re
import shlex
from pathlib import Path

import yaml

import pyrosetta
from pyrosetta.rosetta.core.pose import setPoseExtraScore
from pyrosetta.rosetta.core.select.residue_selector import get_residue_selector
from rosettalink.decorators import register_mover
from rosettalink.utils import get_run_command
from rosettalink.utils import mover_configuration
from rosettalink.utils import run_and_log
from rosettalink.utils import setup_tracer
from rosettalink.utils import work_dir
import rosettalink.movers.BaseLinkMover as BaseLinkMover


# Design name the mover gives a spec it writes itself. rfd4-generate uses it
# as the example_id when it names output files.
EXAMPLE_ID = "design"

# config_file key holding the condition spec rather than a hydra argument.
DESIGNS_KEY = "designs"

# atom_site columns rfd4 records its conditioning in, written in this order,
# and the values they carry where nothing was held. Coordinates are held per
# atom and only for the backbone; a sequence is held per residue.
CONDITION_SEQUENCE = "condition_sequence_1_residue"
CONDITION_SOURCE = "annotation_src-coordinate-conditioned-region_1_atom"
CONDITION_COORDINATES = "condition_CoordinateGroup_1_atom"
MASKED = "<M>"
UNSET = "."
BACKBONE = ("N", "CA", "C", "O")

# A contig segment naming a chain, e.g. B1-150 or B60. The chain is the
# leading letters.
KEPT_SEGMENT = re.compile(r"^([A-Za-z]+)(\d+)(?:-(\d+))?$")

# Metrics csv columns that name a file or repeat the run configuration.
NOT_A_SCORE = {
    "example_id", "designed_structure_path", "designed_chain_path",
    "input_json_path", "output_base_dir", "metrics_tags", "task",
    "dataset_name", "chain_of_interest",
}


class RFD4Proteina(BaseLinkMover.BaseLinkMover):
    clones_ = list()

    def __init__(self, mode="generate", contig=None, num_replicates=None,
                 batch_size=None, hold_input=True, metrics_tags=None,
                 conditioned_selector=None, spec_file=None,
                 config_file=None, run_command="",
                 extra_args=None, work_dir=None, prefix_name=""):
        BaseLinkMover.BaseLinkMover.__init__(self)
        self.mode_ = mode
        self.contig_ = contig
        self.num_replicates_ = num_replicates
        self.batch_size_ = batch_size
        self.hold_input_ = hold_input
        self.metrics_tags_ = metrics_tags
        self.conditioned_selector_ = conditioned_selector
        self.spec_file_ = spec_file
        self.config_file_ = config_file
        self.run_command_ = run_command
        self.extra_args_ = extra_args
        self.work_dir_ = work_dir
        self.prefix_name_ = prefix_name
        self.additional_poses_ = None

        (self.tracer_fatal, self.tracer_error, self.tracer_warning,
         self.tracer_info, self.tracer_debug, self.tracer_trace,
         *_) = setup_tracer("[RFD4Proteina]")

    def clone(self):
        copy = RFD4Proteina()
        copy.mode_ = self.mode_
        copy.contig_ = self.contig_
        copy.num_replicates_ = self.num_replicates_
        copy.batch_size_ = self.batch_size_
        copy.hold_input_ = self.hold_input_
        copy.metrics_tags_ = self.metrics_tags_
        copy.conditioned_selector_ = self.conditioned_selector_
        copy.spec_file_ = self.spec_file_
        copy.config_file_ = self.config_file_
        copy.run_command_ = self.run_command_
        copy.extra_args_ = self.extra_args_
        copy.work_dir_ = self.work_dir_
        copy.prefix_name_ = self.prefix_name_
        RFD4Proteina.clones_.append(copy)
        return copy

    def apply(self, pose):
        run_command = self._run_command()
        input_poses = self.take_input_poses(pose)
        # Read before work_dir() changes directory: a relative path on the tag
        # names the run directory.
        settings, designs = self._read_config_file()
        spec = self._read_spec_file(designs)

        with work_dir(self.work_dir_, self.tracer_debug) as run_dir:
            self.tracer_info << f"Current working directory: {os.getcwd()} \n" and self.tracer_info.flush()
            self.tracer_info << f"{self.mode_} from {len(input_poses)} input pose(s) \n" and self.tracer_info.flush()

            if self.mode_ == "evaluate":
                results = self._evaluate(input_poses, run_command, settings)
            else:
                results = self._generate(input_poses, run_command, settings, spec)

            pose.assign(results[0])
            self.additional_poses_ = list(results[1:])

    def _run_command(self):
        """The command prefix for the entry point mode names."""
        if self.run_command_:
            return self.run_command_

        if self.mode_ == "evaluate":
            configured = mover_configuration(self.mover_name()).get(
                "evaluate_run_command", "")
            if configured:
                return configured

        run_command = get_run_command(self.mover_name(), self.run_command_)
        if self.mode_ == "evaluate":
            return run_command.replace("rfd4-generate", "rfd4-evaluate")
        return run_command

    # ---------------------------------------------------------------- inputs

    def _read_config_file(self):
        """The hydra settings and the condition spec of config_file."""
        if not self.config_file_:
            return {}, None

        path = Path(self.config_file_).resolve()
        if not path.is_file():
            raise Exception(f"RFD4Proteina config_file does not exist: {path}")

        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise Exception(
                f"RFD4Proteina config_file {path} must hold a mapping of "
                f"settings, found {type(loaded).__name__}")

        designs = loaded.pop(DESIGNS_KEY, None)
        if designs is not None and self.spec_file_:
            raise Exception(
                f"RFD4Proteina was given both spec_file and a {DESIGNS_KEY} "
                f"key in config_file {path}. Use one or the other")
        self.tracer_info << f"Read {len(loaded)} setting(s) from {path} \n" and self.tracer_info.flush()
        return loaded, designs

    def _read_spec_file(self, designs):
        """The condition spec to use, or None to build one from contig."""
        if designs is not None:
            return designs
        if not self.spec_file_:
            return None

        path = Path(self.spec_file_).resolve()
        if not path.is_file():
            raise Exception(f"RFD4Proteina spec_file does not exist: {path}")
        return yaml.safe_load(path.read_text())

    def _dump_structure(self, pose, path, conditioned=None):
        """Writes one pose as the mmCIF the program reads, without hydrogens.

        conditioned is a per residue vector of the residues whose sequence and
        coordinates were held while the structure was designed. Given one,
        three further atom_site columns record it the way rfd4 records it in
        its own output."""
        pyrosetta.rosetta.core.io.mmcif.dump_cif(pose, str(path))

        kept, columns, in_atom_site = [], [], False
        residue, previous_key = 0, None
        for line in Path(path).read_text().splitlines():
            if line.startswith("_atom_site."):
                columns.append(line.split(".", 1)[1].strip())
                in_atom_site = True
                kept.append(line)
                continue

            if in_atom_site and line.startswith(("ATOM", "HETATM")):
                fields = line.split()
                if fields[columns.index("type_symbol")] == "H":
                    continue
                if conditioned is not None:
                    # dump_cif writes atoms in pose order, so each new chain
                    # and sequence number is the next residue.
                    key = (fields[columns.index("label_asym_id")],
                           fields[columns.index("label_seq_id")])
                    if key != previous_key:
                        residue += 1
                        previous_key = key
                    held = residue <= len(conditioned) and conditioned[residue - 1]
                    backbone = held and (
                        fields[columns.index("label_atom_id")] in BACKBONE)
                    line = (f"{line} "
                            f"{fields[columns.index('label_comp_id')] if held else MASKED} "
                            f"{''.join(key) if backbone else UNSET} "
                            f"{'0' if backbone else '-1'}")
            elif in_atom_site and line.startswith("#"):
                in_atom_site = False

            kept.append(line)

        if conditioned is not None:
            header = kept.index(f"_atom_site.{columns[-1]}") + 1
            kept[header:header] = [f"_atom_site.{CONDITION_SEQUENCE}",
                                   f"_atom_site.{CONDITION_SOURCE}",
                                   f"_atom_site.{CONDITION_COORDINATES}"]
        Path(path).write_text("\n".join(kept) + "\n")

    def _conditioned_residues(self, pose):
        """Which residues conditioned_selector names, or None when unset."""
        if self.conditioned_selector_ is None:
            return None
        selection = self.conditioned_selector_.apply(pose)
        conditioned = [selection[resnum]
                       for resnum in range(1, pose.total_residue() + 1)]

        held = sum(conditioned)
        if held in (0, len(conditioned)):
            self.tracer_warning << (
                f"conditioned_selector names {held} of {len(conditioned)} "
                f"residue(s), so no part of the structure is left as the "
                f"designed one and the metrics scoped to it are dropped \n"
            ) and self.tracer_warning.flush()
        return conditioned

    def _kept_chains(self):
        """The chains named by a contig segment, in the order they appear."""
        chains = []
        for segment in (self.contig_ or "").split(","):
            match = KEPT_SEGMENT.match(segment.strip())
            if match and match.group(1) not in chains:
                chains.append(match.group(1))
        return chains

    def _conditions(self):
        """Conditions holding the backbone and the sequence of every chain the
        contig keeps."""
        if not self.hold_input_:
            return None
        chains = self._kept_chains()
        if not chains:
            return None
        return {
            "C_CRD": {
                True: [
                    {"select": f"sel('{chain}/*/*/[N,CA,C,O]') and not has_nan_coord()"}
                    for chain in chains
                ]
            },
            "C_SEQ": {
                True: [{"select": f"sel('{chain}')"} for chain in chains]
            },
        }

    def _write_spec(self, spec_path, spec, input_name):
        """Writes the condition spec rfd4-generate is pointed at.

        A given spec is used as it stands, with every design naming no input
        pointed at the structure dumped beside it. Without one a single design
        keyed by EXAMPLE_ID is built from contig."""
        if spec is None:
            design = {"input": input_name}
            if self.contig_:
                design["contig"] = self.contig_
            conditions = self._conditions()
            if conditions:
                design["conditions"] = conditions
                self.tracer_info << f"Holding coordinates and sequence of chain(s): {', '.join(self._kept_chains())} \n" and self.tracer_info.flush()
            spec = {EXAMPLE_ID: design}
        else:
            if not isinstance(spec, dict) or not spec:
                raise Exception(
                    "RFD4Proteina was given a condition spec holding no "
                    "design: a spec is keyed by design name")
            for name, design in spec.items():
                if name != "global_args":
                    design.setdefault("input", input_name)

        with open(spec_path, "w") as handle:
            yaml.safe_dump(spec, handle, sort_keys=False)
        self.tracer_info << f"Wrote condition spec {spec_path}: {list(spec)} \n" and self.tracer_info.flush()
        return spec

    # ------------------------------------------------------------- arguments

    def _arguments(self, settings, inputs_value):
        """The hydra key=value arguments of one invocation."""
        arguments = dict(settings)

        if self.num_replicates_ is not None:
            arguments["num_replicates"] = self.num_replicates_
        if self.batch_size_ is not None:
            arguments["batch_size"] = self.batch_size_
        if self.metrics_tags_:
            arguments["metrics_tags"] = self.metrics_tags_
        tags = arguments.get("metrics_tags")
        if isinstance(tags, str):
            arguments["metrics_tags"] = self._as_list(tags)

        arguments["inputs"] = inputs_value
        arguments["out_dir"] = "output"

        rendered = [f"{key}={shlex.quote(self._as_value(value))}"
                    for key, value in arguments.items()]

        # Generate runs no metrics until something asks for them.
        if (self.mode_ != "evaluate" and "metrics_tags" not in arguments
                and not self._mentions_metrics(settings)):
            rendered.append("'~metrics'")

        if self.extra_args_:
            rendered.append(self.extra_args_)
        return " ".join(rendered)

    @staticmethod
    def _as_list(value):
        """A metrics_tags attribute as hydra's bracketed list."""
        text = str(value).strip()
        if text.startswith("["):
            return text
        tags = [tag for tag in re.split(r"[,\s]+", text) if tag]
        return "[" + ",".join(tags) + "]"

    @staticmethod
    def _as_value(value):
        """One setting as hydra reads it on a command line."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(str(item) for item in value) + "]"
        return str(value)

    @staticmethod
    def _mentions_metrics(settings):
        return any(str(key).startswith("metrics") for key in settings)

    # -------------------------------------------------------------- generate

    def _generate(self, input_poses, run_command, settings, spec):
        """Runs the generate entry point once per input structure and returns
        every design, loaded, labelled and scored."""
        designed_poses = []
        for index, input_pose in enumerate(input_poses):
            if input_pose.total_residue() == 0:
                raise Exception(
                    "RFD4Proteina needs an input structure: a condition spec "
                    "requires an input, and a contig requires at least one "
                    "preserved segment")

            design_dir = Path(f"input_{index}") if len(input_poses) > 1 else Path(".")
            previous_dir = os.getcwd()
            os.makedirs(design_dir, exist_ok=True)
            os.chdir(design_dir)
            try:
                self._dump_structure(input_pose, "input.cif")
                self._write_spec("spec.yaml", spec, "input.cif")
                run_and_log(
                    f"{run_command} {self._arguments(settings, 'spec.yaml')}",
                    self.tracer_info, self.tracer_error)

                scores = self._read_metrics(Path("output/inference_metrics"))
                for cif_file in self._generated_files():
                    designed_poses.append(
                        self._load_design(cif_file, scores, cif_file.parent.name))
            finally:
                os.chdir(previous_dir)
        return designed_poses

    def _generated_files(self):
        """Every design the generate entry point wrote, sorted by name.

        A scored run writes its refold into the same directory, so the search
        keys off the _generated suffix."""
        samples_dir = Path("output/inference_samples")
        cif_files = sorted(samples_dir.rglob("*_generated.cif"))
        if cif_files:
            self.tracer_info << f"Found design(s): {[str(cif) for cif in cif_files]} \n" and self.tracer_info.flush()
            return cif_files

        compressed = sorted(samples_dir.rglob("*_generated.cif.zst"))
        if compressed:
            raise Exception(
                f"Found only Zstandard compressed designs in {samples_dir}, "
                f"which cannot be read into a pose: "
                f"{[str(cif) for cif in compressed]}. Run without "
                f"compress_outputs=true")

        self.tracer_error << f"No designs found in {samples_dir} \n" and self.tracer_error.flush()
        raise Exception(f"No designs found in {samples_dir}")

    # -------------------------------------------------------------- evaluate

    def _evaluate(self, input_poses, run_command, settings):
        """Scores the poses it was handed and returns them carrying the
        metrics, in the order they came in."""
        if not self.metrics_tags_ and "metrics_tags" not in settings:
            raise Exception(
                "RFD4Proteina mode=evaluate needs metrics_tags, which selects "
                "the metrics that run, e.g. metrics_tags=\"ppi_refolding\". "
                "Set it on the tag or in config_file")

        inputs_dir = Path("inputs")
        inputs_dir.mkdir(exist_ok=True)
        names = []
        for index, input_pose in enumerate(input_poses):
            name = f"pose_{index}.cif"
            self._dump_structure(input_pose, inputs_dir / name,
                                 self._conditioned_residues(input_pose))
            names.append(name)

        run_and_log(f"{run_command} {self._arguments(settings, str(inputs_dir))}",
                    self.tracer_info, self.tracer_error)

        scores = self._read_metrics(Path("output/evaluation_metrics"),
                                    key_column="designed_structure_path")
        if not scores:
            raise Exception(
                "RFD4Proteina mode=evaluate produced no metrics. The entry "
                "point skips a structure it cannot parse and still exits 0, "
                "so check the log for parse warnings")

        scored = []
        for index, input_pose in enumerate(input_poses):
            emitted = input_pose.clone()
            row = scores.get(names[index])
            if row is None:
                raise Exception(
                    f"RFD4Proteina mode=evaluate returned no metrics row for "
                    f"{names[index]}: scored {sorted(scores)}")
            self._apply_scores(emitted, row)
            scored.append(emitted)
        return scored

    # ---------------------------------------------------------------- output

    def _read_metrics(self, metrics_dir, key_column="example_id"):
        """Every metrics row the run wrote, keyed by the file name in
        key_column. An absent directory gives no rows and is not an error."""
        if not metrics_dir.is_dir():
            return {}

        rows = {}
        for csv_file in sorted(metrics_dir.glob("*.csv")):
            if csv_file.name.startswith("aggregated"):
                continue
            with open(csv_file, newline="") as handle:
                for row in csv.DictReader(handle):
                    key = row.get(key_column)
                    if key:
                        rows[Path(key).name] = row
        if rows:
            self.tracer_info << f"Read metrics for {len(rows)} structure(s) from {metrics_dir} \n" and self.tracer_info.flush()
        return rows

    def _apply_scores(self, pose, row):
        """Puts every numeric column of one metrics row onto the pose."""
        applied = 0
        for column, value in row.items():
            if column in NOT_A_SCORE or value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            setPoseExtraScore(pose, f"{self.prefix_name_}{column}", number)
            applied += 1
        if applied:
            self.tracer_info << f"\t{applied} metric(s) scored \n" and self.tracer_info.flush()

    def _load_design(self, cif_file, scores, example_id):
        """Loads one design, stamps the `all` reslabel and carries over any
        metrics measured for it."""
        pose = pyrosetta.pose_from_file(str(cif_file))

        label = f"{self.prefix_name_}all"
        for resnum in range(1, pose.total_residue() + 1):
            pose.pdb_info().add_reslabel(resnum, label)
        self.tracer_info << f"\t{label}: {pose.total_residue()} residue(s) \n" and self.tracer_info.flush()

        row = scores.get(example_id)
        if row is not None:
            self._apply_scores(pose, row)
        return pose

    def get_additional_output(self):
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ RFD4Proteina. Self: {self}, tag: {tag}, data: {data} \n" and self.tracer_debug.flush()
        self.mode_ = tag.get_option_string("mode") if tag.hasOption("mode") else "generate"
        if self.mode_ not in ("generate", "evaluate"):
            raise Exception(
                f"RFD4Proteina mode must be generate or evaluate, not {self.mode_}")
        self.contig_ = tag.get_option_string("contig") if tag.hasOption("contig") else None
        self.num_replicates_ = tag.get_option_int("num_replicates") if tag.hasOption("num_replicates") else None
        self.batch_size_ = tag.get_option_int("batch_size") if tag.hasOption("batch_size") else None
        self.hold_input_ = tag.get_option_bool("hold_input") if tag.hasOption("hold_input") else True
        self.metrics_tags_ = tag.get_option_string("metrics_tags") if tag.hasOption("metrics_tags") else None
        self.conditioned_selector_ = (
            get_residue_selector(tag.get_option_string("conditioned_selector"), data)
            if tag.hasOption("conditioned_selector") else None)
        self.spec_file_ = tag.get_option_string("spec_file") if tag.hasOption("spec_file") else None
        self.config_file_ = tag.get_option_string("config_file") if tag.hasOption("config_file") else None
        self.run_command_ = tag.get_option_string("run_command") if tag.hasOption("run_command") else ""
        self.extra_args_ = tag.get_option_string("extra_args") if tag.hasOption("extra_args") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else None
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else ""

        self.tracer_info << (
            f"Parsed options: mode: {self.mode_}, contig: {self.contig_}, "
            f"num_replicates: {self.num_replicates_}, batch_size: "
            f"{self.batch_size_}, metrics_tags: {self.metrics_tags_}, "
            f"config_file: {self.config_file_}, spec_file: {self.spec_file_}, "
            f"run_command: {self.run_command_}, extra_args: {self.extra_args_}, "
            f"work_dir: {self.work_dir_} \n") and self.tracer_info.flush()

    @staticmethod
    def mover_name():
        return "RFD4Proteina"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_integer, xs_boolean

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "mode",
            XMLSchemaType(xs_string),
            "Entry point to run. generate designs structures from the poses it is given, "
            "the default. evaluate scores them instead and emits them unchanged apart "
            "from their score keys. contig, num_replicates, batch_size and hold_input "
            "apply to generate; metrics_tags is required by evaluate",
            "generate"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "contig",
            XMLSchemaType(xs_string),
            "Contig handed to rfd4-generate, comma separated, e.g. A1-120, /0, 50-120. "
            "A42-48 keeps a segment of the input, /0 starts a new chain, 50-120 is an "
            "expandable segment and E1:2-20 a named one. It names chains and residues "
            "of the dumped input.cif, where each chain is numbered from 1. The whole "
            "input is kept when unset, and it is unused when a condition spec is given",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "num_replicates",
            XMLSchemaType(xs_integer),
            "Number of designs generated per input structure. Left to the program's own "
            "default when unset",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "batch_size",
            XMLSchemaType(xs_integer),
            "Number of designs processed at once on the GPU. Raises memory use without "
            "changing how many designs are produced. Left to the program's own default "
            "when unset",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "hold_input",
            XMLSchemaType(xs_boolean),
            "Condition the model on the backbone and the sequence of every chain the "
            "contig keeps, true by default. With it false a kept target comes back "
            "refolded and resequenced rather than preserved",
            "true"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "metrics_tags",
            XMLSchemaType(xs_string),
            "Tags selecting which metrics run, space or comma separated, e.g. "
            "ppi_refolding or unconditional_refolding. Required by mode=evaluate; in "
            "generate mode scoring stays off unless this is set. Every numeric column "
            "of the metrics the run writes is put on the pose as a score key",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "conditioned_selector",
            XMLSchemaType(xs_string),
            "Residues whose sequence and coordinates were held while the structure was "
            "designed, named as a residue selector, for mode=evaluate. Recorded in the "
            "structure the way rfd4 records it in its own output, which is what names "
            "the designed chain to the metrics scoped to it and to the refolding oracle. "
            "Without it that choice is ambiguous on a structure rfd4 did not generate",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "config_file",
            XMLSchemaType(xs_string),
            "One yaml file holding the whole run. Every top-level key becomes a hydra "
            "key=value argument, so num_replicates, generation, seed and the rest can be "
            "written there instead of on the tag. The reserved key designs holds the "
            "condition spec. Attributes set on the tag win over it, and extra_args over "
            "both",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "spec_file",
            XMLSchemaType(xs_string),
            "Condition spec yaml on its own, keyed by design name, each design taking "
            "input, contig and conditions. Every design naming no input is pointed at "
            "the dumped pose. Use this or the designs key of config_file, not both",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "run_command",
            XMLSchemaType(xs_string),
            "Command that runs rfd4-generate, e.g. singularity run --nv /path/to/rfd4.sif "
            "rfd4-generate, or /path/to/env/bin/rfd4-generate. Taken from the RFD4Proteina "
            "section of rosettalink.config.yaml when not set here. With mode=evaluate the "
            "token rfd4-generate in it is replaced with rfd4-evaluate",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "extra_args",
            XMLSchemaType(xs_string),
            "Extra hydra arguments, space separated, e.g. seed=42 "
            "generation=inverse_folding. Appended last. Paths are relative to the run "
            "directory",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "prefix_name",
            XMLSchemaType(xs_string),
            "Put in front of every reslabel and score key this mover stamps, so "
            "prefix_name=rfd4_ gives rfd4_all. Empty by default, which leaves them "
            "unprefixed",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Directory the run happens in. A fresh temporary directory is used when "
            "unset. Do not set it to an empty string, which pyrosetta reports as an "
            "invalid option",
            ""))

        description = '''
                        Runs RFD4-Proteina to generate backbone designs, or to
                        score structures it is given.
                      '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd,
            cls.mover_name(),
            description, attrlist)


@register_mover
class RFD4ProteinaCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = RFD4Proteina()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return RFD4Proteina.mover_name()

    def provide_xml_schema(self, xsd):
        RFD4Proteina.provide_xml_schema(xsd)
