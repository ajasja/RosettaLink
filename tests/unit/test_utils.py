from pathlib import Path

from rosettalink.utils import parse_fasta_records


def test_parse_fasta_records():
	fasta_path = Path(__file__).with_name("basic.fasta")
	records = parse_fasta_records(fasta_path)

	assert records == [
		("prot1", ["AAAAAAAAA", "DDDDDDDDD"]),
		("prot2", ["AAAAAAAAA", "DDDDDDDDD"]),
		("prot2_2", ["AGHAGHAGH"]),
	]
