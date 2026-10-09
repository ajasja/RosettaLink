import os

import pytest

from rosettalink import utils


@pytest.fixture(autouse=True)
def empty_configuration():
	utils.configuration.clear()
	yield
	utils.configuration.clear()


def write_config(directory, text):
	path = directory / utils.CONFIG_FILENAME
	path.write_text(text)
	return path


def test_explicit_path_is_loaded(tmp_path):
	path = write_config(tmp_path, "ColabFold:\n  run_command: colabfold_batch\n")

	loaded_from = utils.load_configuration(str(path))

	assert loaded_from == str(path)
	assert utils.configuration["ColabFold"]["run_command"] == "colabfold_batch"


def test_missing_explicit_path_is_an_error(tmp_path):
	with pytest.raises(FileNotFoundError):
		utils.load_configuration(str(tmp_path / "nope.yaml"))


def test_current_directory_wins_over_home(tmp_path, monkeypatch):
	home = tmp_path / "home"
	(home / ".rosettalink").mkdir(parents=True)
	write_config(home / ".rosettalink", "ColabFold:\n  run_command: from_home\n")

	working = tmp_path / "working"
	working.mkdir()
	write_config(working, "ColabFold:\n  run_command: from_cwd\n")

	monkeypatch.setattr(utils.Path, "home", classmethod(lambda cls: home))
	monkeypatch.chdir(working)

	utils.load_configuration()

	assert utils.configuration["ColabFold"]["run_command"] == "from_cwd"


def test_home_is_used_when_there_is_none_in_the_current_directory(tmp_path, monkeypatch):
	home = tmp_path / "home"
	(home / ".rosettalink").mkdir(parents=True)
	write_config(home / ".rosettalink", "ColabFold:\n  run_command: from_home\n")

	working = tmp_path / "working"
	working.mkdir()

	monkeypatch.setattr(utils.Path, "home", classmethod(lambda cls: home))
	monkeypatch.chdir(working)

	utils.load_configuration()

	assert utils.configuration["ColabFold"]["run_command"] == "from_home"


def test_no_configuration_anywhere_leaves_it_empty(tmp_path, monkeypatch):
	monkeypatch.setattr(utils.Path, "home", classmethod(lambda cls: tmp_path / "home"))
	monkeypatch.chdir(tmp_path)

	assert utils.load_configuration() is None
	assert utils.configuration == {}


def test_a_configuration_that_is_not_a_mapping_is_rejected(tmp_path, monkeypatch):
	write_config(tmp_path, "- just\n- a list\n")
	monkeypatch.setattr(utils.Path, "home", classmethod(lambda cls: tmp_path / "home"))
	monkeypatch.chdir(tmp_path)

	with pytest.raises(RuntimeError):
		utils.load_configuration()


def test_loading_replaces_the_previous_configuration(tmp_path):
	first = tmp_path / "first.yaml"
	first.write_text("ColabFold:\n  run_command: first\n")
	second = tmp_path / "second.yaml"
	second.write_text("RFdiffusion:\n  run_command: second\n")

	utils.load_configuration(str(first))
	utils.load_configuration(str(second))

	assert "ColabFold" not in utils.configuration
	assert utils.configuration["RFdiffusion"]["run_command"] == "second"


def test_run_command_from_the_tag_wins():
	utils.configuration.update({"ColabFold": {"run_command": "from_config"}})

	assert utils.get_run_command("ColabFold", "from_tag") == "from_tag"


def test_run_command_falls_back_to_the_configuration():
	utils.configuration.update({"ColabFold": {"run_command": "from_config"}})

	assert utils.get_run_command("ColabFold", "") == "from_config"


def test_run_command_without_either_names_the_mover_and_the_file():
	with pytest.raises(RuntimeError) as error:
		utils.get_run_command("ColabFold", "")

	message = str(error.value)
	assert "ColabFold" in message
	assert utils.CONFIG_FILENAME in message


def test_a_section_that_is_not_a_mapping_is_ignored():
	utils.configuration.update({"ColabFold": "colabfold_batch"})

	with pytest.raises(RuntimeError):
		utils.get_run_command("ColabFold", "")
