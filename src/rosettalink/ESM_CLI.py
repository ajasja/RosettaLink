import argparse
from Bio import SeqIO # To read fasta file
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--fasta_file",
    type=str,
    required=True,
    help="Path to the input FASTA file containing protein sequences (chains split by colon).")
parser.add_argument(
    "--msa",
    type=str,
    required=False,
    help="Path to the input MSA file in .a3m format (optional). The first sequence must be the query.")
parser.add_argument(
    "--msa_max_sequences",
    type=int,
    default=None,
    help="Maximum number of MSA sequences to load from the .a3m file (default: all).")
parser.add_argument(
    "--kernel",
    type=str,
    required=False,
    help="Help"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run the model on (default: 'cuda').")
parser.add_argument(
    "--model",
    type=str,
    default="biohub/ESMFold2-Fast",
    help="Pretrained model to use (default: 'biohub/ESMFold2-Fast').")
parser.add_argument(
    "--output_prefix",
    type=str,
    default="output_",
    help="Output filename prefix (default: 'output_').")
parser.add_argument(
    "--weights",
    type=str,
    default=None,
    help="Path to model weights cache (HF_HUB_CACHE env var)."
)
parser.add_argument(
    "--online",
    action="store_true",
    help="Download new model weights from Hugging Face Hub (do not set HF_HUB_OFFLINE=1). (Default: False)"
)



args = parser.parse_args()

if args.weights:
    os.environ["HF_HUB_CACHE"] = args.weights
if not args.online:
    os.environ["HF_HUB_OFFLINE"] = "1"

if args.msa and args.model == "biohub/ESMFold2-Fast":
    print(f"WARNING: You are trying to use MSA with model {args.model}, which is not supported. The non-fast `ESMFold2` model should be used instead. ")

def read_fasta_file(fasta_file):
    with open(fasta_file, "r") as file_handle:
        sequences = list(SeqIO.parse(file_handle, "fasta"))
    if not sequences:
        raise ValueError(f"No sequences found in the FASTA file: {fasta_file}")
    return sequences

# Read fasta first, so exit early if bad seq
sequences = read_fasta_file(args.fasta_file)

# Must be imported after seting HF_HOME/HF_HUB_CACHE env var
from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
from esm.utils.msa import MSA

print("After importing esm and msa?")

def match_msa_to_sequence(msa, sequence, record_id):
    """Return an MSA whose columns line up with the (possibly colon-joined) sequence.

    ESMFold2 splits ':'-joined sequences into chains and slices the MSA by position,
    skipping one column per colon. A concatenated MSA without separator columns
    (e.g. ColabFold paired .a3m) gets a placeholder column inserted at each chain break.
    """
    chains = sequence.split(":")
    if len(chains) > 1 and msa.seqlen == len(sequence) - (len(chains) - 1):
        indices, pos = [], 0
        for chain in chains:
            if indices:
                indices.append(pos - 1)  # Placeholder column, discarded by the model
            indices.extend(range(pos, pos + len(chain)))
            pos += len(chain)
        msa = msa.select_positions(indices)
    if msa.seqlen != len(sequence):
        raise ValueError(f"MSA length ({msa.seqlen}) does not match length of sequence {record_id} ({len(sequence)}, or {len(sequence.replace(':', ''))} without colons)")
    return msa

# Load MSA before the model, so exit early if it doesn't match the query
msas = [None] * len(sequences)
if args.msa:
    msa = MSA.from_a3m(path=args.msa, remove_insertions=True, max_sequences=args.msa_max_sequences)
    print(f"MSA depth: {msa.depth} sequences, length: {msa.seqlen} residues")
    for i, record in enumerate(sequences):
        msas[i] = match_msa_to_sequence(msa, str(record.seq), record.id)
        query = "".join(q for q, s in zip(msas[i].sequences[0], str(record.seq)) if s != ":")
        if query != str(record.seq).replace(":", ""):
            print(f"WARNING: First MSA sequence does not match query {record.id}")

model = EsmFold2Model.from_pretrained(args.model, device=args.device).eval()
builder = ESMFold2InputBuilder()

for i, record in enumerate(sequences):
    print("-----")
    print(record)
    print(f"ID: {record.id}")
    print(f"Sequence: {record.seq}")
    print(f"Description: {record.description}")
    outname = f"{args.output_prefix}{i}.cif"

    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=str(record.seq), msa=msas[i])])
    result = builder.fold(model, spi, num_loops=10, num_sampling_steps=100, seed=0)

    print(f"pLDDT {float(result.plddt.mean()):.3f}   pTM {float(result.ptm):.3f}")
    with open(outname, "w") as f:
        f.write(result.complex.to_mmcif())

