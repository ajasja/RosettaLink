"""Argument parsing of the integration test stubs.

A stub that misreads its command line writes the wrong file layout, and the
integration test then fails for a reason that has nothing to do with the
mover under test. These cover the parsing only, so they need neither
PyRosetta nor a fixture.
"""

import json
import sys
from pathlib import Path

import pytest

STUBS = Path(__file__).absolute().parents[1] / "integration" / "stubs"
sys.path.insert(0, str(STUBS))

import colabfold as colabfold_stub
import ligandmpnn as ligandmpnn_stub
import replay
import rfdiffusion as rfdiffusion_stub


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #


def test_an_option_value_is_the_argument_that_follows_it():
	assert replay.option(["--out_folder", ".", "--batch_size", "4"], "--batch_size") == "4"


def test_an_absent_option_takes_the_default():
	assert replay.option(["--out_folder", "."], "--batch_size", "1") == "1"


def test_a_trailing_option_without_a_value_takes_the_default():
	assert replay.option(["--batch_size"], "--batch_size", "1") == "1"


def test_emit_cycles_when_more_is_asked_for_than_was_recorded(tmp_path):
	source = tmp_path / "recorded.pdb"
	source.write_text("ATOM\n")
	targets = [tmp_path / f"out_{n}.pdb" for n in range(3)]

	replay.emit([source], targets)

	assert all(target.read_text() == "ATOM\n" for target in targets)


# --------------------------------------------------------------------------- #
# RFdiffusion: hydra key=value arguments
# --------------------------------------------------------------------------- #


def test_a_hydra_setting_is_read_from_key_equals_value():
	argv = ["inference.output_prefix=output/", "inference.num_designs=3", "-cd", "output"]

	assert rfdiffusion_stub.setting(argv, "inference.num_designs") == "3"
	assert rfdiffusion_stub.setting(argv, "inference.output_prefix") == "output/"


def test_a_hydra_setting_keeps_everything_after_the_first_equals():
	argv = ["contigmap.contigs=[B1-150/0 90-120]"]

	assert rfdiffusion_stub.setting(argv, "contigmap.contigs") == "[B1-150/0 90-120]"


def test_an_absent_hydra_setting_takes_the_default():
	assert rfdiffusion_stub.setting([], "inference.num_designs", "1") == "1"


# --------------------------------------------------------------------------- #
# LigandMPNN: one input, or a json of many
# --------------------------------------------------------------------------- #


def test_a_single_input_comes_from_pdb_path():
	assert ligandmpnn_stub.input_stems(["--pdb_path", "input.pdb"]) == ["input"]


def test_several_inputs_come_from_the_pdb_path_multi_json(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	multi = tmp_path / "pdb_path_multi.json"
	multi.write_text(json.dumps({"design_0.pdb": "", "design_1.pdb": ""}))

	assert ligandmpnn_stub.input_stems(["--pdb_path_multi", str(multi)]) == [
		"design_0",
		"design_1",
	]


def test_an_invocation_without_any_input_is_an_error():
	with pytest.raises(replay.StubError):
		ligandmpnn_stub.input_stems(["--out_folder", "."])


# --------------------------------------------------------------------------- #
# ColabFold: two positional arguments after the options
# --------------------------------------------------------------------------- #


def test_the_fasta_and_output_directory_are_the_last_positional_arguments():
	argv = ["--model-order", "1", "--msa-mode", "single_sequence", "--rank", "auto",
	        "input.fasta", "."]

	fasta, out_dir = colabfold_stub.positional(argv)

	assert fasta == Path("input.fasta")
	assert out_dir == Path(".")


def test_an_invocation_without_both_positionals_is_an_error():
	with pytest.raises(replay.StubError):
		colabfold_stub.positional(["--rank", "auto", "input.fasta"])


def test_record_ids_carry_the_header_and_the_total_length(tmp_path):
	fasta = tmp_path / "input.fasta"
	fasta.write_text(">design_0\nAAAA:CCCC\n>design_1\nGGGG\n")

	# The chain separator is not a residue, so the complex is 8 long.
	assert colabfold_stub.record_ids(fasta) == [("design_0", 8), ("design_1", 4)]


def test_a_missing_fasta_is_an_error(tmp_path):
	with pytest.raises(replay.StubError):
		colabfold_stub.record_ids(tmp_path / "absent.fasta")


def test_a_prediction_keeps_the_model_that_won_the_recorded_ranking():
	recorded = Path("design_0_unrelaxed_rank_001_alphafold2_multimer_v3_model_2_seed_000.pdb")

	assert colabfold_stub.renamed(recorded, "design_3", "_unrelaxed_rank_001") == (
		"design_3_unrelaxed_rank_001_alphafold2_multimer_v3_model_2_seed_000.pdb"
	)


def test_a_record_id_containing_an_underscore_is_replaced_whole():
	recorded = Path("design_0_scores_rank_001_alphafold2_ptm_model_1_seed_000.json")

	assert colabfold_stub.renamed(recorded, "design_10", "_scores_rank_001").startswith(
		"design_10_scores_rank_001"
	)


def test_the_recording_of_the_same_record_is_preferred():
	recorded = {"design_0": 100, "design_1": 250}

	assert colabfold_stub.choose(recorded, "design_1", 250, 0) == "design_1"


def test_the_same_name_does_not_win_against_the_right_length():
	# A wider fan out than the recording reuses names for designs of a
	# different length, so the name alone is not enough.
	recorded = {"design_0": 249, "design_1": 250}

	assert colabfold_stub.choose(recorded, "design_1", 249, 0) == "design_0"


def test_an_unrecorded_record_takes_a_recording_of_its_own_length():
	# A prediction of the wrong length cannot stand in for this one: the RMSD
	# metrics select on both structures and need them to match.
	recorded = {"design_0": 249, "design_1": 250}

	assert colabfold_stub.choose(recorded, "design_5", 250, 0) == "design_1"


def test_a_length_no_recording_has_falls_back_in_order():
	recorded = {"design_0": 249}

	assert colabfold_stub.choose(recorded, "design_9", 400, 0) == "design_0"
