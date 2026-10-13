"""Every attribute an example XML sets on a RosettaLink mover tag is one that
mover actually declares.

An attribute that no longer exists is rejected by
XmlObjects.create_from_string() with "is not a valid option", which costs a
Rosetta startup to discover and a GPU allocation if it happens inside a
submitted job. The schema is read with ast, so this runs without PyRosetta.
"""

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from rosettalink.scripts.run_xml import PLACEHOLDER

ROOT = Path(__file__).absolute().parents[2]
MOVER_FILES = sorted((ROOT / "src" / "rosettalink" / "movers").glob("*.py"))
XML_FILES = sorted((ROOT / "examples").glob("*.xml"))

# Rosetta adds these to every mover tag itself.
ALWAYS_ALLOWED = {"name"}


def parsed(path):
	return ast.parse(path.read_text(), filename=str(path))


def mover_name(tree):
	"""The string returned by the module's mover_name(), i.e. its tag name."""
	for node in ast.walk(tree):
		if not isinstance(node, ast.FunctionDef) or node.name != "mover_name":
			continue
		for statement in ast.walk(node):
			if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Constant):
				return statement.value.value
	return None


def declared_attributes(tree):
	"""Attribute names the module declares, from the XMLSchemaAttribute calls
	in provide_xml_schema() and from the OPTION_HELP table of the movers that
	pass their wrapped tool's options straight through."""
	names = set()
	for node in ast.walk(tree):
		if (
			isinstance(node, ast.Call)
			and isinstance(node.func, ast.Attribute)
			and node.func.attr in ("required_attribute", "attribute_w_default", "attribute")
			and node.args
			and isinstance(node.args[0], ast.Constant)
		):
			names.add(node.args[0].value)

	for node in tree.body:
		if isinstance(node, ast.Assign) and any(
			isinstance(target, ast.Name) and target.id == "OPTION_HELP"
			for target in node.targets
		):
			names.update(
				key.value for key in node.value.keys if isinstance(key, ast.Constant)
			)
	return names


def mover_schemas():
	"""{tag name: set of attributes} for every mover in the package."""
	schemas = {}
	for path in MOVER_FILES:
		tree = parsed(path)
		name = mover_name(tree)
		if name:
			schemas[name] = declared_attributes(tree) | ALWAYS_ALLOWED
	return schemas


SCHEMAS = mover_schemas()


def mover_tags(path):
	"""Every element of an example XML whose name is a RosettaLink mover,
	paired with the mover it belongs to. A subtag such as the RMSD of a
	predictor declares its attributes in the same schema, so it is checked
	against its parent."""
	text = PLACEHOLDER.sub("placeholder", path.read_text())
	found = []
	for element in ET.fromstring(text).iter():
		if element.tag not in SCHEMAS:
			continue
		found.append((element.tag, element))
		for child in element:
			if child.tag not in SCHEMAS:
				found.append((element.tag, child))
	return found


def test_the_movers_were_found():
	# A rename that this test cannot follow would otherwise make it pass by
	# checking nothing at all.
	assert "ColabFold" in SCHEMAS
	assert "run_command" in SCHEMAS["ColabFold"]


@pytest.mark.parametrize("path", XML_FILES, ids=lambda path: path.name)
def test_every_attribute_used_in_an_example_is_declared(path):
	unknown = []
	for mover, element in mover_tags(path):
		for attribute in element.attrib:
			if attribute not in SCHEMAS[mover]:
				unknown.append(f"{element.tag}/{attribute} (not declared by {mover})")

	assert not unknown, f"{path.name} sets attributes no mover declares: {unknown}"


@pytest.mark.parametrize("path", XML_FILES, ids=lambda path: path.name)
def test_no_attribute_is_set_to_an_empty_string(path):
	# An empty attribute value is reported by Rosetta as "not a valid option",
	# which points at the attribute rather than at its value.
	empty = [
		f"{element.tag}/{attribute}"
		for _, element in mover_tags(path)
		for attribute, value in element.attrib.items()
		if value == ""
	]
	assert not empty, f"{path.name} sets empty attribute values: {empty}"
