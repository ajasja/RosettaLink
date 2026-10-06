"""Argument handling and protocol filling of the rosetta_link_scripts command.

Only the parts that do not need PyRosetta; main() itself imports it.
"""

import json

import pytest

from rosettalink.scripts import run_xml


def parse(argv):
	parser = run_xml.build_parser()
	return parser.parse_known_args(run_xml.translate_legacy(argv))


def test_long_options():
	arguments, unknown = parse(["--protocol", "design.xml", "--input", "1ubq.pdb"])

	assert arguments.protocol == "design.xml"
	assert arguments.input == ["1ubq.pdb"]
	assert unknown == []


def test_rosetta_scripts_options():
	arguments, unknown = parse(
		["-parser:protocol", "design.xml", "-s", "1ubq.pdb", "-out:prefix", "min_", "-nstruct", "3"]
	)

	assert arguments.protocol == "design.xml"
	assert arguments.input == ["1ubq.pdb"]
	assert arguments.out_prefix == "min_"
	assert arguments.nstruct == 3
	assert unknown == []


def test_several_input_structures():
	arguments, _ = parse(["--protocol", "design.xml", "-s", "a.pdb", "b.pdb", "c.pdb"])

	assert arguments.input == ["a.pdb", "b.pdb", "c.pdb"]


def test_unrecognised_options_are_left_for_pyrosetta():
	arguments, unknown = parse(
		["--protocol", "design.xml", "-mute", "all", "-ignore_unrecognized_res"]
	)

	assert arguments.protocol == "design.xml"
	assert unknown == ["-mute", "all", "-ignore_unrecognized_res"]


def test_scorefile_format_is_checked():
	with pytest.raises(SystemExit):
		parse(["--protocol", "design.xml", "--scorefile-format", "sqlite"])


def test_the_rosetta_scripts_spelling_of_a_var():
	arguments, _ = parse(["--protocol", "p.xml", "-parser:script_vars", "num=4"])

	assert run_xml.parse_vars(arguments.var) == {"num": "4"}


def test_the_short_spelling_of_a_var():
	# XML forbids '--' inside a comment, so an example command written in one
	# can only use this spelling.
	arguments, _ = parse(["--protocol", "p.xml", "-var", "num=4"])

	assert run_xml.parse_vars(arguments.var) == {"num": "4"}


def test_vars_are_parsed():
	assert run_xml.parse_vars(["contig=[100-100]", "num=4"]) == {
		"contig": "[100-100]",
		"num": "4",
	}


def test_a_var_without_a_value_is_rejected():
	with pytest.raises(SystemExit):
		run_xml.parse_vars(["contig"])


def test_a_var_value_may_contain_an_equals_sign():
	assert run_xml.parse_vars(["args=a=1,b=2"]) == {"args": "a=1,b=2"}


def test_placeholders_are_filled():
	filled = run_xml.fill_protocol(
		'<RFdiffusion contig="{contig}" num_designs="{num}" />',
		{"contig": "[100-100]", "num": "4"},
	)

	assert filled == '<RFdiffusion contig="[100-100]" num_designs="4" />'


def test_an_unfilled_placeholder_is_reported_by_name():
	with pytest.raises(SystemExit) as error:
		run_xml.fill_protocol('<RFdiffusion contig="{contig}" />', {})

	assert "contig" in str(error.value)


def test_a_protocol_without_placeholders_needs_no_vars():
	xml = '<RFdiffusion contig="[100-100]" />'

	assert run_xml.fill_protocol(xml, {}) == xml


def test_json_scorefile(tmp_path):
	rows = [{"description": "a.pdb", "AF2_plddt": 90.1}]

	path = run_xml.write_scorefile(rows, tmp_path / "score", "json")

	assert json.loads(path.read_text()) == rows


def test_csv_scorefile_columns_are_the_union_of_every_row(tmp_path):
	rows = [
		{"description": "a.pdb", "AF2_plddt": 90.1},
		{"description": "b.pdb", "AF2_scRMSD_designed": 1.2},
	]

	path = run_xml.write_scorefile(rows, tmp_path / "score", "csv")

	header = path.read_text().splitlines()[0]
	assert header.strip() == "description,AF2_plddt,AF2_scRMSD_designed"


def test_no_scorefile_is_written_when_asked_for_none(tmp_path):
	assert run_xml.write_scorefile([{"a": 1}], tmp_path / "score", "none") is None
	assert list(tmp_path.iterdir()) == []
