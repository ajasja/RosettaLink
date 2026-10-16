"""The ESMFold2 mover against a stand-in for ESM_CLI.py.

The stub checks the fasta the mover wrote and answers with the input trimer,
two of its chains exchanged, in the file layout ESM_CLI.py writes. Needs
PyRosetta, but no GPU and no esm.
"""

import sys

import pytest

pytest.importorskip("pyrosetta", reason="the integration tests need PyRosetta")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIMER = ROOT / "examples" / "input_data" / "coiled_coil_trimer.pdb"

STUB = '''
import json, shutil, sys
from pathlib import Path

argv = sys.argv[1:]
fasta = Path(argv[argv.index("--fasta_file") + 1]).read_text().split()
prefix = argv[argv.index("--output_prefix") + 1]
assert "--num_loops" in argv, argv
assert fasta[0] == ">design_0" and fasta[1].count(":") == 2, fasta
length = len(fasta[1].replace(":", ""))
shutil.copy(Path(__file__).with_name("prediction.cif"), f"{prefix}0.cif")
Path(f"{prefix}0_scores.json").write_text(json.dumps({
    "plddt": [0.5] * length, "pae": [[2.0] * length] * length, "ptm": 0.8, "iptm": None,
}))
'''


@pytest.fixture(scope="module")
def folded(tmp_path_factory):
	import pyrosetta
	import rosettalink

	rosettalink.init("-mute all")
	from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

	tmp_path = tmp_path_factory.mktemp("esmfold2")
	trimer = pyrosetta.pose_from_file(str(TRIMER))
	parts = trimer.split_by_chain()
	swapped = pyrosetta.Pose()
	for index in (1, 3, 2):
		pyrosetta.rosetta.core.pose.append_pose_to_pose(swapped, parts[index], True)
	# Rosetta refuses to superimpose identical coordinates, so move the stand-in.
	for resnum in range(1, swapped.total_residue() + 1):
		residue = swapped.residue(resnum)
		for atom in range(1, residue.natoms() + 1):
			xyz = residue.xyz(atom)
			nudge = 0.1 if resnum == 1 and atom == 2 else 0.0
			swapped.set_xyz(
				pyrosetta.AtomID(atom, resnum),
				pyrosetta.rosetta.numeric.xyzVector_double_t(xyz.x + 5.0 + nudge, xyz.y, xyz.z),
			)
	pyrosetta.rosetta.core.io.mmcif.dump_cif(swapped, str(tmp_path / "prediction.cif"))
	(tmp_path / "stub.py").write_text(STUB)

	objects = XmlObjects.create_from_string(f"""
	<ROSETTASCRIPTS>
	  <RESIDUE_SELECTORS>
	    <Chain name="a" chains="A"/>
	  </RESIDUE_SELECTORS>
	  <MOVERS>
	    <ESMFold2 name="fold" run_command="{sys.executable} {tmp_path / 'stub.py'}"
	              extra_args="--num_loops 3">
	      <RMSD name="rmsd"/>
	      <metric name="plddt_a" base_metric="pLDDT" selector="a"/>
	    </ESMFold2>
	  </MOVERS>
	  <PROTOCOLS></PROTOCOLS>
	</ROSETTASCRIPTS>
	""")
	pose = trimer.clone()
	objects.get_mover("fold").apply(pose)
	return dict(pose.scores)


def test_confidence_scores_are_attached_and_missing_ones_skipped(folded):
	assert folded["ESMFold2_plddt"] == pytest.approx(0.5)
	assert folded["ESMFold2_pae"] == pytest.approx(2.0)
	assert folded["ESMFold2_ptm"] == pytest.approx(0.8)
	assert "ESMFold2_iptm" not in folded
	assert folded["ESMFold2_plddt_a"] == pytest.approx(0.5)


def test_the_rmsd_pairs_exchanged_chains_back(folded):
	assert folded["ESMFold2_rmsd"] == pytest.approx(0.0, abs=0.01)


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
