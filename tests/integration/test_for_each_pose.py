"""Whether the installed PyRosetta understands FOR_EACH_POSE.

The syntax is not in a release. It needs a build carrying
RosettaCommons/rosetta pull request 494, and a build without it fails at
parse time with

    Element 'FOR_EACH_POSE': This element is not expected.

Every protocol that scores each design of a fan-out depends on it, so this
says plainly which of those a build can run. No GPU, no container, no
external program, and no RosettaLink mover: only the parser is under test.
"""

import pytest

pytest.importorskip("pyrosetta", reason="the integration tests need PyRosetta")


CONTROL = """
<ROSETTASCRIPTS>
  <FILTERS><NetCharge name="netcharge" confidence="0"/></FILTERS>
  <PROTOCOLS><Add filter="netcharge"/></PROTOCOLS>
</ROSETTASCRIPTS>
"""

FOR_EACH_POSE = """
<ROSETTASCRIPTS>
  <FILTERS><NetCharge name="netcharge" confidence="0"/></FILTERS>
  <PROTOCOLS>
    <FOR_EACH_POSE><PROTOCOLS><Add filter="netcharge"/></PROTOCOLS></FOR_EACH_POSE>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""

PROTOCOLS_SUBTAG = """
<ROSETTASCRIPTS>
  <FILTERS><NetCharge name="netcharge" confidence="0"/></FILTERS>
  <PROTOCOLS>
    <MultiplePoseMover name="each">
      <PROTOCOLS><Add filter="netcharge"/></PROTOCOLS>
    </MultiplePoseMover>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""


def parse(xml):
	import rosettalink
	from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

	rosettalink.init("-mute all")
	XmlObjects.create_from_string(xml)


def test_the_build_parses_an_ordinary_protocol():
	# The control. A failure here is not about FOR_EACH_POSE.
	parse(CONTROL)


def test_the_build_parses_for_each_pose():
	parse(FOR_EACH_POSE)


def test_the_build_parses_a_protocols_subtag_on_multipleposemover():
	# The other half of pull request 494: MultiplePoseMover taking PROTOCOLS
	# directly rather than a nested ROSETTASCRIPTS block.
	parse(PROTOCOLS_SUBTAG)


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
