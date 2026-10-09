"""Static checks over every mover source, from docs/writing_movers.md section 8.

These read the source with ast rather than importing it, so they run without
PyRosetta and catch the mistakes that otherwise only surface mid-job:

  - an attribute missing from clone() is silently dropped the moment
    RosettaScripts clones the mover
  - '<', '>' or '&' in a schema description string makes Rosetta reject its
    own XSD, for every protocol rather than only ones using that mover
"""

import ast
from pathlib import Path

import pytest

MOVERS_DIR = Path(__file__).absolute().parents[2] / "src" / "rosettalink" / "movers"
MOVER_FILES = sorted(MOVERS_DIR.glob("*.py"))

# Attributes deliberately absent from clone(): per-apply() state, not options.
NOT_CLONED = {"additional_poses_"}

FORBIDDEN_IN_DESCRIPTIONS = ("<", ">", "&")


def mover_class(tree):
	"""The Mover subclass of a module, i.e. the class defining mover_name()."""
	for node in ast.walk(tree):
		if not isinstance(node, ast.ClassDef):
			continue
		names = {child.name for child in node.body if isinstance(child, ast.FunctionDef)}
		if "mover_name" in names and "apply" in names:
			return node
	return None


def method(class_node, name):
	for child in class_node.body:
		if isinstance(child, ast.FunctionDef) and child.name == name:
			return child
	return None


def self_attributes_assigned(function_node):
	"""Names like foo_ from every `self.foo_ = ...` in a function."""
	found = set()
	for node in ast.walk(function_node):
		if not isinstance(node, ast.Assign):
			continue
		for target in node.targets:
			if (
				isinstance(target, ast.Attribute)
				and isinstance(target.value, ast.Name)
				and target.value.id == "self"
				and target.attr.endswith("_")
			):
				found.add(target.attr)
	return found


def copy_attributes_assigned(function_node):
	"""Names from every `copy.foo_ = ...` in clone()."""
	found = set()
	for node in ast.walk(function_node):
		if not isinstance(node, ast.Assign):
			continue
		for target in node.targets:
			if (
				isinstance(target, ast.Attribute)
				and isinstance(target.value, ast.Name)
				and target.value.id == "copy"
			):
				found.add(target.attr)
	return found


def parsed(path):
	return ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", MOVER_FILES, ids=lambda path: path.name)
def test_mover_source_is_valid_python(path):
	parsed(path)


@pytest.mark.parametrize("path", MOVER_FILES, ids=lambda path: path.name)
def test_every_option_set_in_init_is_also_copied_by_clone(path):
	class_node = mover_class(parsed(path))
	if class_node is None:
		pytest.skip(f"{path.name} defines no mover")
	clone = method(class_node, "clone")
	if clone is None:
		pytest.skip(f"{class_node.name} does not override clone()")

	from_init = self_attributes_assigned(method(class_node, "__init__")) - NOT_CLONED
	# Tracers are rebuilt by the constructor of the copy.
	from_init = {name for name in from_init if not name.startswith("tracer_")}
	from_clone = copy_attributes_assigned(clone)

	missing = sorted(from_init - from_clone)
	assert not missing, f"{class_node.name}.clone() does not copy: {missing}"


@pytest.mark.parametrize("path", MOVER_FILES, ids=lambda path: path.name)
def test_every_option_set_by_parse_my_tag_is_also_copied_by_clone(path):
	class_node = mover_class(parsed(path))
	if class_node is None:
		pytest.skip(f"{path.name} defines no mover")
	clone = method(class_node, "clone")
	parse_my_tag = method(class_node, "parse_my_tag")
	if clone is None or parse_my_tag is None:
		pytest.skip(f"{class_node.name} does not override both")

	from_tag = self_attributes_assigned(parse_my_tag) - NOT_CLONED
	missing = sorted(from_tag - copy_attributes_assigned(clone))
	assert not missing, f"{class_node.name}.clone() does not copy: {missing}"


@pytest.mark.parametrize("path", MOVER_FILES, ids=lambda path: path.name)
def test_no_angle_brackets_or_ampersands_in_schema_strings(path):
	class_node = mover_class(parsed(path))
	if class_node is None:
		pytest.skip(f"{path.name} defines no mover")
	schema = method(class_node, "provide_xml_schema")
	if schema is None:
		pytest.skip(f"{class_node.name} provides no schema")

	offenders = []
	for node in ast.walk(schema):
		if isinstance(node, ast.Constant) and isinstance(node.value, str):
			if any(character in node.value for character in FORBIDDEN_IN_DESCRIPTIONS):
				offenders.append(node.value)

	assert not offenders, (
		f"{class_node.name} schema strings may not contain '<', '>' or '&': {offenders}"
	)
