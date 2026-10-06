"""Every example XML is well formed once its placeholders are filled.

Parsing an XML string is free, running it is not: a stray '--' inside a
comment or an unbalanced tag fails at XmlObjects.create_from_string(), before
any mover runs, so it costs nothing to catch here.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from rosettalink.scripts.run_xml import PLACEHOLDER

EXAMPLES_DIR = Path(__file__).absolute().parents[2] / "examples"
XML_FILES = sorted(EXAMPLES_DIR.glob("*.xml"))


@pytest.mark.parametrize("path", XML_FILES, ids=lambda path: path.name)
def test_example_xml_is_well_formed(path):
	text = path.read_text()
	for name in set(PLACEHOLDER.findall(text)):
		text = text.replace("{" + name + "}", "placeholder")

	ET.fromstring(text)


@pytest.mark.parametrize("path", XML_FILES, ids=lambda path: path.name)
def test_no_double_hyphen_inside_an_xml_comment(path):
	# XML forbids '--' inside a comment, which rules out writing the long
	# options of rosetta_link_scripts in one.
	for comment in ET.ElementTree(ET.fromstring(
		PLACEHOLDER.sub("placeholder", path.read_text())
	)).getroot().iter(ET.Comment):
		assert "--" not in str(comment.text)
