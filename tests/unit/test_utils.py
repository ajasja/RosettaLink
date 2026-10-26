import os
from pathlib import Path

import pytest

from rosettalink.utils import RMSD_ATOM_SETS
from rosettalink.utils import SENTINEL_NAME
from rosettalink.utils import drain_additional_output
from rosettalink.utils import forget_finished_work_dirs
from rosettalink.utils import is_reusing_finished_output
from rosettalink.utils import parse_fasta_records
from rosettalink.utils import run_and_log
from rosettalink.utils import work_dir


class FakeTracer:
	"""Stands in for a Rosetta Tracer: collects what is written to it."""

	def __init__(self):
		self.messages = []

	def __lshift__(self, text):
		self.messages.append(text)
		return self

	def flush(self):
		pass


# --------------------------------------------------------------------------- #
# parse_fasta_records
# --------------------------------------------------------------------------- #


def test_parse_fasta_records():
	fasta_path = Path(__file__).with_name("basic.fasta")
	records = parse_fasta_records(fasta_path)

	assert records == [
		("prot1", ["AAAAAAAAA", "DDDDDDDDD"]),
		("prot2", ["AAAAAAAAA", "DDDDDDDDD"]),
		("prot2_2", ["AGHAGHAGH"]),
	]


def test_a_sequence_split_over_several_lines_is_one_chain(tmp_path):
	fasta = tmp_path / "wrapped.fasta"
	fasta.write_text(">one\nAAAA\nCCCC\nGGGG\n")

	assert parse_fasta_records(fasta) == [("one", ["AAAACCCCGGGG"])]


def test_whitespace_inside_a_sequence_is_ignored(tmp_path):
	fasta = tmp_path / "spaced.fasta"
	fasta.write_text(">one\nAAA AAA : CCC\n")

	assert parse_fasta_records(fasta) == [("one", ["AAAAAA", "CCC"])]


def test_only_the_first_token_of_a_header_names_the_record(tmp_path):
	fasta = tmp_path / "described.fasta"
	fasta.write_text(">design_7 some description here\nAAAA\n")

	assert parse_fasta_records(fasta) == [("design_7", ["AAAA"])]


def test_characters_a_file_name_cannot_hold_are_replaced(tmp_path):
	fasta = tmp_path / "awkward.fasta"
	fasta.write_text(">chain|A/1\nAAAA\n")

	record_id, _ = parse_fasta_records(fasta)[0]
	assert record_id == "chain_A_1"


def test_an_empty_fasta_is_rejected(tmp_path):
	fasta = tmp_path / "empty.fasta"
	fasta.write_text("")

	with pytest.raises(RuntimeError, match="No sequences found"):
		parse_fasta_records(fasta)


def test_a_record_without_a_sequence_is_skipped(tmp_path):
	fasta = tmp_path / "partial.fasta"
	fasta.write_text(">empty\n>real\nAAAA\n")

	assert parse_fasta_records(fasta) == [("real", ["AAAA"])]


# --------------------------------------------------------------------------- #
# work_dir
# --------------------------------------------------------------------------- #


def test_work_dir_chdirs_into_the_directory_it_yields(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)

	with work_dir() as run_dir:
		assert Path(os.getcwd()).samefile(run_dir)


def test_the_original_directory_is_restored(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	before = os.getcwd()

	with work_dir():
		pass

	assert os.getcwd() == before


def test_the_original_directory_is_restored_after_an_exception(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	before = os.getcwd()

	with pytest.raises(RuntimeError):
		with work_dir():
			raise RuntimeError("the wrapped program failed")

	assert os.getcwd() == before


def test_the_temporary_directory_is_removed(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)

	with work_dir() as run_dir:
		Path("output.pdb").write_text("ATOM\n")

	assert not run_dir.exists()


def test_the_run_happens_in_work_dir(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "kept"

	with work_dir(str(kept)) as run_dir:
		assert Path(run_dir).samefile(kept)
		Path("output.pdb").write_text("ATOM\n")
		os.makedirs("backbones")
		Path("backbones/design_1.pdb").write_text("ATOM\n")
		assert (kept / "output.pdb").is_file()

	assert (kept / "output.pdb").read_text() == "ATOM\n"
	assert (kept / "backbones" / "design_1.pdb").is_file()


def test_work_dir_is_created_when_it_does_not_exist(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "missing" / "nested"

	with work_dir(str(kept)):
		Path("output.pdb").write_text("ATOM\n")

	assert (kept / "output.pdb").is_file()


def test_a_relative_work_dir_resolves_against_the_directory_it_started_in(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)

	with work_dir("kept"):
		Path("output.pdb").write_text("ATOM\n")

	assert (tmp_path / "kept" / "output.pdb").is_file()


def test_a_second_run_overwrites_the_output_of_the_first(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "kept"

	with work_dir(str(kept)):
		Path("output.pdb").write_text("first\n")
	with work_dir(str(kept)):
		Path("output.pdb").write_text("second\n")

	assert (kept / "output.pdb").read_text() == "second\n"


def test_nothing_is_kept_without_a_work_dir(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)

	with work_dir():
		Path("output.pdb").write_text("ATOM\n")

	assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Resuming a finished stage
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _forget_finished():
	forget_finished_work_dirs()
	yield
	forget_finished_work_dirs()


def test_a_finished_run_leaves_the_sentinel(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "kept"

	with work_dir(str(kept)):
		Path("output.pdb").write_text("ATOM\n")

	assert (kept / SENTINEL_NAME).is_file()


def test_a_failed_run_leaves_no_sentinel(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "kept"

	with pytest.raises(RuntimeError):
		with work_dir(str(kept)):
			raise RuntimeError("the wrapped program failed")

	assert not (kept / SENTINEL_NAME).exists()


def test_no_sentinel_without_a_work_dir(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)

	with work_dir() as run_dir:
		assert not (run_dir / SENTINEL_NAME).exists()

	assert list(tmp_path.iterdir()) == []


def test_a_later_run_reuses_a_finished_work_dir(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "kept"

	with work_dir(str(kept)):
		assert not is_reusing_finished_output()
		Path("output.pdb").write_text("ATOM\n")

	forget_finished_work_dirs()
	with work_dir(str(kept)):
		assert is_reusing_finished_output()


def test_a_second_apply_in_one_run_does_not_reuse(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "kept"

	with work_dir(str(kept)):
		pass
	with work_dir(str(kept)):
		assert not is_reusing_finished_output()


def test_the_flag_is_off_again_after_the_block(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "kept"

	with work_dir(str(kept)):
		pass
	forget_finished_work_dirs()
	with work_dir(str(kept)):
		assert is_reusing_finished_output()

	assert not is_reusing_finished_output()


def test_run_and_log_raises_on_a_failing_command(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	tracer = FakeTracer()

	with work_dir(str(tmp_path / "kept")):
		with pytest.raises(Exception):
			run_and_log("exit 7", tracer, tracer)


def test_a_reused_stage_does_not_run_its_command(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	kept = tmp_path / "kept"
	tracer = FakeTracer()

	with work_dir(str(kept)):
		pass

	forget_finished_work_dirs()
	with work_dir(str(kept)):
		run_and_log("exit 7", tracer, tracer)

	assert any(SENTINEL_NAME in message for message in tracer.messages)


# --------------------------------------------------------------------------- #
# drain_additional_output
# --------------------------------------------------------------------------- #


class FakeMover:
	"""A one-to-many mover: one pose per call, None once exhausted."""

	def __init__(self, poses):
		self.poses = list(poses)

	def get_additional_output(self):
		if not self.poses:
			return None
		return self.poses.pop(0)


def test_drain_collects_every_pose_until_none():
	assert drain_additional_output(FakeMover(["a", "b", "c"])) == ["a", "b", "c"]


def test_drain_of_a_mover_with_nothing_more_is_empty():
	assert drain_additional_output(FakeMover([])) == []


def test_drain_leaves_the_mover_exhausted():
	mover = FakeMover(["a", "b"])
	drain_additional_output(mover)

	assert drain_additional_output(mover) == []


# --------------------------------------------------------------------------- #
# RMSD atom sets
# --------------------------------------------------------------------------- #


def test_every_short_name_maps_to_an_rmsd_atoms_enum_name():
	# resolve_rmsd_atoms() looks each up on core::scoring::rmsd_atoms, so a
	# name not of that form cannot resolve.
	assert all(value.startswith("rmsd_") for value in RMSD_ATOM_SETS.values())


def test_ca_is_the_alpha_carbon_atom_set():
	assert RMSD_ATOM_SETS["ca"] == "rmsd_protein_bb_ca"


# --------------------------------------------------------------------------- #
# work_dir resolution
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_output_root():
	"""Leaves output_root as it was found, so one test cannot change where
	another one's work_dir lands."""
	import rosettalink.utils as utils

	previous = utils.output_root
	utils.set_output_root(None)
	yield utils
	utils.output_root = previous


def test_a_relative_work_dir_lands_under_the_output_root(no_output_root, tmp_path):
	no_output_root.set_output_root(tmp_path)

	assert no_output_root.resolve_work_dir("01_rfdiff") == tmp_path / "01_rfdiff"


def test_an_absolute_work_dir_is_left_alone(no_output_root, tmp_path):
	no_output_root.set_output_root(tmp_path)
	elsewhere = tmp_path.parent / "elsewhere"

	assert no_output_root.resolve_work_dir(elsewhere) == elsewhere


def test_without_an_output_root_a_relative_work_dir_follows_the_current_directory(
	no_output_root, tmp_path, monkeypatch
):
	monkeypatch.chdir(tmp_path)

	assert no_output_root.resolve_work_dir("01_rfdiff") == (tmp_path / "01_rfdiff").absolute()


def test_no_work_dir_resolves_to_nothing(no_output_root):
	assert no_output_root.resolve_work_dir(None) is None
	assert no_output_root.resolve_work_dir("") is None
