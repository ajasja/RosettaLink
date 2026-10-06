"""Shared setup for the integration tests.

These run a whole protocol through rosetta_link_scripts with the external
programs replaced by the stubs in stubs/, which replay the output of a real
run recorded under fixtures/. They need PyRosetta but no GPU and no wrapped
tool, so they exercise everything RosettaLink does between one program and
the next: command construction, output discovery, reslabels, the pose buffer,
score keys and the RMSD metrics.

Record a new fixture by running the same protocol for real with work_dir set
to the fixture directory; see readme.md.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[2]
STUBS = Path(__file__).absolute().parent / "stubs"
FIXTURES = Path(__file__).absolute().parent / "fixtures"

pytest.importorskip("pyrosetta", reason="the integration tests need PyRosetta")

MOVER_OF = {
	"rfdiffusion": "RFdiffusion",
	"ligandmpnn": "LigandMPNN",
	"colabfold": "ColabFold",
}


def stub_configuration(path, tools):
	"""Writes a configuration whose run_command reaches a stub instead of the
	real program. The stubs run under the interpreter running the tests."""
	lines = []
	for tool in tools:
		lines.append(f"{MOVER_OF[tool]}:")
		lines.append(f"  run_command: {sys.executable} {STUBS / tool}.py")
	path.write_text("\n".join(lines) + "\n")
	return path


@pytest.fixture(scope="module")
def run_protocol(tmp_path_factory):
	"""Runs one protocol against stubbed programs and returns its output
	directory together with the rows of its score file.

	Module scoped: a protocol is run once and every test of that module reads
	the same result, rather than paying for a Rosetta startup each.
	"""

	def run(fixture, tools, protocol, inputs, variables=None):
		missing = [tool for tool in tools if not (FIXTURES / fixture / tool).is_dir()]
		if missing:
			pytest.skip(f"no recorded {fixture} fixture for {missing}")

		from rosettalink.scripts.run_xml import main
		from rosettalink.utils import pose_buffer

		tmp_path = tmp_path_factory.mktemp(fixture)
		output_dir = tmp_path / "output"
		config = stub_configuration(tmp_path / "rosettalink.config.yaml", tools)

		argv = ["--protocol", protocol, "--output-dir", str(output_dir), "--config", str(config)]
		for path in inputs:
			argv += ["--input", path]
		for name, value in (variables or {}).items():
			argv += ["--var", f"{name}={value}"]

		previous_dir = os.getcwd()
		previous_fixture = os.environ.get("ROSETTALINK_STUB_FIXTURE")
		os.environ["ROSETTALINK_STUB_FIXTURE"] = str(FIXTURES / fixture)
		os.chdir(ROOT)
		pose_buffer.clear()
		try:
			main(argv)
		finally:
			os.chdir(previous_dir)
			pose_buffer.clear()
			if previous_fixture is None:
				os.environ.pop("ROSETTALINK_STUB_FIXTURE", None)
			else:
				os.environ["ROSETTALINK_STUB_FIXTURE"] = previous_fixture

		return output_dir, json.loads((output_dir / "score.json").read_text())

	return run
