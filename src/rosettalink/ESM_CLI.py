# Uses model params as per HF_HOME env var:
# export HF_HOME=.../shared_folder/software/hf_cache

import argparse
from Bio import SeqIO # To read fasta file
from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput


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
    help="Path to the input MSA file (optional).")
parser.add_argument(
    "--kernel",
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


args = parser.parse_args()

def read_fasta_file(fasta_file):
    with open(fasta_file, "r") as file_handle:
        sequences = list(SeqIO.parse(file_handle, "fasta"))
    if not sequences:
        raise ValueError(f"No sequences found in the FASTA file: {fasta_file}")
    return sequences

# Read fasta first, so exit early if bad seq
sequences = read_fasta_file(args.fasta_file)

model = EsmFold2Model.from_pretrained(args.model, device=args.device).eval()
builder = ESMFold2InputBuilder()

for i, record in enumerate(sequences):
    print("-----")
    print(record)
    print(f"ID: {record.id}")
    print(f"Sequence: {record.seq}")
    print(f"Description: {record.description}")
    outname = f"{args.output_prefix}{i}.cif"

    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=str(record.seq))])
    result = builder.fold(model, spi, num_loops=10, num_sampling_steps=100, seed=0)

    print(f"pLDDT {float(result.plddt.mean()):.3f}   pTM {float(result.ptm):.3f}")
    with open(outname, "w") as f:
        f.write(result.complex.to_mmcif())

