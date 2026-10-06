# @file scripts/run_xml.py
# @brief Runs any RosettaScripts XML over any number of input structures.
#
# Installed as the rosetta_link_scripts command. It takes the rosetta_scripts
# spellings of the options it supports as well as long ones:
#
#   rosetta_link_scripts -parser:protocol design.xml -s 1ubq.pdb -out:prefix min_
#   rosetta_link_scripts --protocol design.xml --input 1ubq.pdb --nstruct 4
#
# Anything it does not recognise is passed to pyrosetta.init(), so Rosetta
# options this script has no opinion about still reach Rosetta:
#
#   rosetta_link_scripts --protocol design.xml -s 1ubq.pdb -mute all
#
# A protocol holding {placeholders} is filled from --var:
#
#   rosetta_link_scripts --protocol design.xml --var contig=[100-100] --var num=4
#   rosetta_link_scripts -parser:protocol design.xml -parser:script_vars num=4
#
# Every structure the protocol produces is written out, the primary pose and
# everything reached through get_additional_output(), together with a score
# file of the values cached on each.

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Option spellings of the rosetta_scripts application, mapped onto the long
# options below. Anything absent here is left for pyrosetta.init().
LEGACY_OPTIONS = {
    "-parser:protocol": "--protocol",
    "-s": "--input",
    "-l": "--input",
    "-in:file:s": "--input",
    "-out:prefix": "--out-prefix",
    "-out:suffix": "--out-suffix",
    "-out:path": "--output-dir",
    "-out:path:all": "--output-dir",
    "-out:nstruct": "--nstruct",
    "-nstruct": "--nstruct",
    "-out:file:scorefile": "--scorefile",
    "-scorefile": "--scorefile",
    "-out:file:scorefile_format": "--scorefile-format",
    "-scorefile_format": "--scorefile-format",
    # One assignment each, unlike rosetta_scripts, which takes several after
    # a single -parser:script_vars.
    "-parser:script_vars": "--var",
    "-var": "--var",
}

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def translate_legacy(argv):
    """Rewrites the rosetta_scripts spellings into the long options argparse
    knows. Everything else is passed through untouched."""
    return [LEGACY_OPTIONS.get(argument, argument) for argument in argv]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rosetta_link_scripts",
        description="Run a RosettaScripts XML over one or more input structures.",
    )
    parser.add_argument("--protocol", required=True, help="RosettaScripts XML to run")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[],
        help="Input structures. Without any, the protocol starts from an empty pose",
    )
    parser.add_argument("--nstruct", type=int, default=1, help="Times to run the protocol per input structure")
    parser.add_argument("--output-dir", default=".", help="Directory the output structures are written to")
    parser.add_argument("--out-prefix", default="", help="Prefix of every output file name")
    parser.add_argument("--out-suffix", default="", help="Suffix of every output file name, before the extension")
    parser.add_argument("--scorefile", default="score", help="Score file name, without extension")
    parser.add_argument(
        "--scorefile-format",
        default="json",
        choices=("json", "csv", "none"),
        help="Score file format",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fill a {NAME} placeholder in the XML. Repeatable",
    )
    parser.add_argument("--config", default=None, help="Path to a rosettalink.config.yaml")
    parser.add_argument(
        "--init-options",
        default="-ex1 -ex2aro",
        help="Options passed to pyrosetta.init()",
    )
    return parser


def parse_vars(items):
    variables = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"--var expects NAME=VALUE, got: {item}")
        variables[name.strip()] = value
    return variables


def fill_protocol(xml_text, variables):
    """Substitutes {NAME} placeholders and reports any left unfilled, since an
    unfilled one fails later inside Rosetta with a less specific message."""
    for name, value in variables.items():
        xml_text = xml_text.replace("{" + name + "}", value)

    unfilled = sorted(set(PLACEHOLDER.findall(xml_text)))
    if unfilled:
        raise SystemExit(
            f"Placeholder(s) left unfilled in the protocol: {', '.join(unfilled)}. "
            f"Pass each as --var NAME=VALUE"
        )
    return xml_text


def pose_scores(pose):
    """Values cached on a pose by the filters, simple metrics and movers that
    ran over it."""
    cache = getattr(pose, "cache", None)
    if cache is not None and hasattr(cache, "fast_items"):
        return dict(cache.fast_items())
    return dict(pose.scores)


def write_scorefile(rows, path, scorefile_format):
    if scorefile_format == "none" or not rows:
        return None
    if scorefile_format == "json":
        path = path.with_suffix(".json")
        with open(path, "w") as handle:
            json.dump(rows, handle, indent=2)
        return path

    path = path.with_suffix(".csv")
    columns = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    parser = build_parser()
    arguments, unknown = parser.parse_known_args(translate_legacy(argv or sys.argv[1:]))

    # Whatever this script has no option for is Rosetta's to interpret.
    extra_options = " ".join(unknown)

    import rosettalink
    rosettalink.init(
        arguments.init_options,
        extra_options=extra_options,
        config=arguments.config,
    )

    import pyrosetta
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
    from rosettalink.utils import drain_additional_output, pose_buffer

    protocol_path = Path(arguments.protocol)
    if not protocol_path.is_file():
        raise SystemExit(f"Protocol not found: {protocol_path}")
    xml_text = fill_protocol(protocol_path.read_text(), parse_vars(arguments.var))

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = [Path(path) for path in arguments.input]
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"Input structure not found: {path}")

    rows = []
    # One job is one input structure run once. The protocol is rebuilt per job
    # so no mover carries state from the previous one.
    jobs = [(path, run) for path in (inputs or [None]) for run in range(arguments.nstruct)]
    for path, run in jobs:
        protocol = XmlObjects.create_from_string(xml_text).get_mover("ParsedProtocol")

        # Poses published by the previous job are not inputs to this one.
        pose_buffer.clear()

        if path is None:
            pose = pyrosetta.Pose()
            stem = "from_protocol"
        else:
            pose = pyrosetta.pose_from_file(str(path))
            stem = path.stem
        if arguments.nstruct > 1:
            stem = f"{stem}_{run:04d}"

        print(f"=== {stem}: applying {protocol_path.name} ===")
        protocol.apply(pose)
        results = [pose] + drain_additional_output(protocol) + pose_buffer.drain()
        print(f"=== {stem}: {len(results)} output structure(s) ===")

        for index, result in enumerate(results):
            name = f"{arguments.out_prefix}{stem}_{index:04d}{arguments.out_suffix}.pdb"
            result.dump_pdb(str(output_dir / name))
            row = {"description": name, "input": str(path) if path else ""}
            row.update(pose_scores(result))
            rows.append(row)
            print(f"\t{name}")

    scorefile = write_scorefile(rows, output_dir / arguments.scorefile, arguments.scorefile_format)
    if scorefile:
        print(f"=== Scores written to {scorefile} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
