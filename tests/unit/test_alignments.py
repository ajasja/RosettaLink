"""The a3m assembly behind ColabFold's msa_mode="partial".

These are pure string functions, so they run without PyRosetta and without
ColabFold. What they have to get right is the mapping: a hit covers the
residues existing_msa_selector picks, every other column is a gap, and the
columns line up with the concatenated sequence the predictor is given.
"""

import pytest

from rosettalink.utils import build_partial_a3m
from rosettalink.utils import read_a3m
from rosettalink.utils import without_insertions


ALIGNMENT = ">query\nMKV\n>hit_one\nMKaV\n>hit_two\nM-V\n"


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def test_read_a3m_returns_the_query_and_the_hits():
	query, hits = read_a3m(ALIGNMENT)

	assert query == "MKV"
	assert hits == [(">hit_one", "MKaV"), (">hit_two", "M-V")]


def test_read_a3m_skips_a_colabfold_cardinality_line():
	query, hits = read_a3m("#3\t1\n" + ALIGNMENT)

	assert query == "MKV"
	assert len(hits) == 2


def test_read_a3m_rejects_an_empty_alignment():
	with pytest.raises(ValueError):
		read_a3m("\n#3\t1\n")


def test_without_insertions_drops_lowercase_columns():
	# Lowercase means an insertion relative to the query, which has no column
	# of its own and so cannot be placed against the sequence.
	assert without_insertions("MKaaV") == "MKV"


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def records_of(a3m):
	"""The assembled file as [(header, sequence)], ignoring the cardinality line."""
	lines = a3m.splitlines()
	assert lines[0].startswith("#")
	return [
		(lines[index], lines[index + 1])
		for index in range(1, len(lines) - 1, 2)
	]


def test_the_cardinality_line_gives_chain_lengths_and_one_copy_each():
	a3m = build_partial_a3m(["MKV", "QQQQ"], [None, None], [None, None])

	assert a3m.splitlines()[0] == "#3,4\t1,1"


def test_each_chain_is_gapped_outside_its_own_span():
	a3m = build_partial_a3m(["MKV", "QQQQ"], [None, None], [None, None])
	records = records_of(a3m)

	assert records[0] == (">A\tB", "MKVQQQQ")
	assert records[1] == (">A", "MKV----")
	assert records[2] == (">B", "---QQQQ")


def test_a_hit_lands_on_the_columns_of_its_own_chain():
	a3m = build_partial_a3m(["MKV", "QQQQ"], [ALIGNMENT, None], [None, None])
	hits = [record for record in records_of(a3m) if record[0].startswith(">hit")]

	# Chain A occupies the first three columns of seven, so a hit of chain A
	# fills those and gaps the rest.
	assert hits == [(">hit_one", "MKV----"), (">hit_two", "M-V----")]


def test_an_uncovered_residue_becomes_a_gap():
	# existing_msa_selector picks only positions 1 and 2 of a four residue
	# chain, so the two hit columns land there and the rest stay gaps.
	a3m = build_partial_a3m(["AQRA"], [">q\nQR\n>hit\nQR\n"], [[1, 2]])
	hits = [record for record in records_of(a3m) if record[0].startswith(">hit")]

	assert hits == [(">hit", "-QR-")]


def test_insertions_are_removed_before_a_hit_is_placed():
	a3m = build_partial_a3m(["AQRA"], [">q\nQR\n>hit\nQxxR\n"], [[1, 2]])
	hits = [record for record in records_of(a3m) if record[0].startswith(">hit")]

	assert hits == [(">hit", "-QR-")]


def test_a_chain_without_an_alignment_contributes_no_hits():
	a3m = build_partial_a3m(["MKV", "QQQQ"], [None, None], [None, None])

	assert not [record for record in records_of(a3m) if record[0].startswith(">hit")]


def test_a_query_that_does_not_match_what_it_covers_is_rejected():
	# Silently accepting this shifts every hit against the sequence, which
	# reads as a plausible but wrong alignment.
	with pytest.raises(ValueError, match="query columns"):
		build_partial_a3m(["MKV"], [ALIGNMENT], [[0, 1]])


def test_every_record_is_as_long_as_the_concatenated_sequence():
	a3m = build_partial_a3m(
		["MKV", "AQRA"], [ALIGNMENT, ">q\nQR\n>hit\nQR\n"], [None, [1, 2]]
	)
	records = records_of(a3m)

	assert {len(sequence) for _, sequence in records} == {len("MKV") + len("AQRA")}


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-v"]))
