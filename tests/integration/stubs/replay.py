"""Shared behaviour of the stub programs.

Each stub stands in for one external tool. It is reached exactly as the real
tool is, through run_command, and writes the file layout the mover expects by
replaying a directory recorded from a real run.

The fixture to replay comes from the ROSETTALINK_STUB_FIXTURE environment
variable. Within it, one subdirectory per tool holds that tool's recorded
output tree.
"""

import os
import shutil
import sys
from pathlib import Path


class StubError(RuntimeError):
	pass


def fixture_dir(tool):
	"""The recorded output tree of one tool."""
	root = os.environ.get("ROSETTALINK_STUB_FIXTURE")
	if not root:
		raise StubError(
			"ROSETTALINK_STUB_FIXTURE is not set, so there is no recorded run to replay"
		)
	path = Path(root) / tool
	if not path.is_dir():
		raise StubError(f"No recorded output for {tool} in {root}")
	return path


def option(argv, name, default=None):
	"""The value following --name, or default."""
	if name not in argv:
		return default
	index = argv.index(name)
	if index + 1 >= len(argv):
		return default
	return argv[index + 1]


def copy_recorded(fixture, skip=()):
	"""Copies the recorded tree into the current directory, leaving alone the
	files the mover itself wrote there."""
	for source in sorted(fixture.rglob("*")):
		relative = source.relative_to(fixture)
		if relative.parts[0] in skip or relative.name in skip:
			continue
		target = Path(relative)
		if source.is_dir():
			target.mkdir(parents=True, exist_ok=True)
		else:
			target.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(source, target)


def recorded_files(fixture, subdirectory, pattern):
	"""The recorded files of one output directory, sorted, as a cycle source."""
	found = sorted((fixture / subdirectory).glob(pattern))
	if not found:
		raise StubError(f"No {pattern} recorded in {fixture / subdirectory}")
	return found


def emit(sources, targets):
	"""Writes one output per requested name, cycling through the recorded
	files when more are asked for than were recorded."""
	for index, target in enumerate(targets):
		target = Path(target)
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(sources[index % len(sources)], target)


def log(message):
	print(f"[stub] {message}", file=sys.stderr)
