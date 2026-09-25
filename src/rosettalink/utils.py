import os
from pyrosetta.rosetta.basic import Tracer, TracerPriority

def run_and_log(command, tracer_info, tracer_error):
    """Runs a command using os.system and also logs the command before running using print"""
    tracer_info << f"Running command: {command} \n" and tracer_info.flush()
    stat = os.system(command)
    wife = os.WIFEXITED(stat)
    exitCode = os.waitstatus_to_exitcode(stat)
    tracer_info << f"Command exited with status {stat} and WIFEXITED {wife}. Exit code: {exitCode} \n" and tracer_info.flush()
    if exitCode != 0:
        tracer_error << f" There was an error running the command. We consider it fatal to prevent any file loss. Check the logs and contact the developer. \n" and tracer_error.flush()
        dodatek = ""

        raise Exception(f" Command exited with exit code {exitCode}\n\n{dodatek}")

def setup_tracer(mover_name):
    new_tracer_fatal = Tracer(mover_name, TracerPriority.t_fatal)
    new_tracer_error = Tracer(mover_name, TracerPriority.t_error)
    new_tracer_warning = Tracer(mover_name, TracerPriority.t_warning)
    new_tracer_info = Tracer(mover_name, TracerPriority.t_info)
    new_tracer_debug = Tracer(mover_name, TracerPriority.t_debug)
    new_tracer_trace = Tracer(mover_name, TracerPriority.t_trace)
    return new_tracer_fatal, new_tracer_error, new_tracer_warning, new_tracer_info, new_tracer_debug, new_tracer_trace



def parse_fasta_records(fasta_path, chain_separator=":"):
    """Reads a fasta into [(record_id, [sequence, ...]), ...]. One record is
    one model, and its sequence is split on chain_separator into chains, so
    "SEQA:SEQB" is a two chain complex.

    record_id is the first whitespace-delimited token of the header with
    anything outside letters, digits, dash and underscore replaced, and a
    suffix added if two records would otherwise share a name, so it is safe
    to use in a file name."""
    records = []
    used_ids = {}
    header = None
    chunks = []

    def flush():
        if header is None:
            return
        chains = [
            part for part in "".join(chunks).replace(" ", "").split(chain_separator) if part
        ]
        if not chains:
            return
        record_id = header
        if record_id in used_ids:
            used_ids[record_id] += 1
            record_id = f"{record_id}_{used_ids[record_id]}"
        else:
            used_ids[record_id] = 1
        records.append((record_id, chains))

    with open(fasta_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                name = line[1:].split()[0] if line[1:].split() else f"seq_{len(records) + 1}"
                header = re.sub(r"[^A-Za-z0-9_-]", "_", name)
                chunks = []
            else:
                chunks.append(line)
    flush()

    if not records:
        raise RuntimeError(f"No sequences found in {fasta_path}")
    return records
