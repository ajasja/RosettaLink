"""Stands in for the boltz CLI.

	python boltz.py PREDICTION_PDB INPUT_DIR --out_dir OUT_DIR [...]

Writes one prediction per yaml in INPUT_DIR, each a copy of PREDICTION_PDB,
under OUT_DIR/boltz_results/predictions/<entry>/, with the confidence json
boltz writes beside it. Every other argument is ignored.
"""

import json
import shutil
import sys
from pathlib import Path

SCORES = {
	"confidence_score": 0.9,
	"complex_plddt": 0.8,
	"ptm": 0.7,
	"iptm": 0.6,
}


def main(argv):
	prediction = Path(argv[1])
	input_dir = Path(argv[2])
	out_dir = Path(argv[argv.index("--out_dir") + 1])

	entries = sorted(path.stem for path in input_dir.glob("*.yaml"))
	if not entries:
		print(f"no yaml entries under {input_dir}", file=sys.stderr)
		return 1

	for entry in entries:
		directory = out_dir / "boltz_results" / "predictions" / entry
		directory.mkdir(parents=True, exist_ok=True)
		stem = f"{entry}_model_0"
		shutil.copyfile(prediction, directory / f"{stem}.pdb")
		(directory / f"confidence_{stem}.json").write_text(json.dumps(SCORES))
		print(f"wrote {stem}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv))
