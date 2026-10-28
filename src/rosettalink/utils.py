import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

# pyrosetta is imported inside the functions that need it, so this module can
# be imported - and its configuration, pose-buffer, work_dir and fasta helpers
# tested - in an environment without PyRosetta.


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
    from pyrosetta.rosetta.basic import Tracer, TracerPriority

    new_tracer_fatal = Tracer(mover_name, TracerPriority.t_fatal)
    new_tracer_error = Tracer(mover_name, TracerPriority.t_error)
    new_tracer_warning = Tracer(mover_name, TracerPriority.t_warning)
    new_tracer_info = Tracer(mover_name, TracerPriority.t_info)
    new_tracer_debug = Tracer(mover_name, TracerPriority.t_debug)
    new_tracer_trace = Tracer(mover_name, TracerPriority.t_trace)
    return new_tracer_fatal, new_tracer_error, new_tracer_warning, new_tracer_info, new_tracer_debug, new_tracer_trace


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Name of the configuration file looked for when no path is given.
CONFIG_FILENAME = "rosettalink.config.yaml"

# Contents of the loaded configuration file. One key per mover name, each
# holding that mover's settings:
#
#   RFdiffusion:
#     run_command: singularity run --nv /path/to/rfdiffusion.sif
#   ColabFold:
#     run_command: colabfold_batch
#
# Populated by load_configuration(), which rosettalink.init() calls. Mutated
# in place, so `from rosettalink.utils import configuration` stays valid.
configuration = {}


def config_search_paths(config=None):
    """Paths load_configuration() looks at, in order. With config given that
    is the only path; otherwise the current directory is tried before
    ~/.rosettalink/."""
    if config:
        return [Path(config)]
    return [
        Path.cwd() / CONFIG_FILENAME,
        Path.home() / ".rosettalink" / CONFIG_FILENAME,
    ]


def load_configuration(config=None):
    """Loads a configuration file into `configuration` and returns the path it
    came from together with every path looked at, so a caller reporting that
    none was found does not walk them a second time.

    Raises FileNotFoundError when config names a file that does not exist, so
    a mistyped path is reported rather than silently ignored."""
    import yaml

    paths = config_search_paths(config)
    for path in paths:
        if not path.is_file():
            continue
        with open(path) as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"{path} must hold a mapping of mover name to settings, got {type(loaded).__name__}"
            )
        configuration.clear()
        configuration.update(loaded)
        return str(path), paths

    if config:
        raise FileNotFoundError(f"Configuration file not found: {config}")
    configuration.clear()
    return None, paths


def mover_configuration(mover_name):
    """Settings of one mover from the loaded configuration, as a dict. Empty
    when the file has no section for it."""
    section = configuration.get(mover_name)
    return section if isinstance(section, dict) else {}


def get_run_command(mover_name, run_command=""):
    """The command prefix that reaches the external program of mover_name.

    The value given on the tag wins; otherwise it comes from the run_command
    key of that mover's section in the configuration file. Raises naming both
    when neither is set."""
    if run_command:
        return run_command

    from_config = mover_configuration(mover_name).get("run_command", "")
    if from_config:
        return from_config

    searched = ", ".join(str(path) for path in config_search_paths())
    raise RuntimeError(
        f"No run_command for {mover_name}. Either set run_command on the tag:\n"
        f'    <{mover_name} name="..." run_command="..." />\n'
        f"or add a section to {CONFIG_FILENAME}:\n"
        f"    {mover_name}:\n"
        f"      run_command: ...\n"
        f"Configuration searched: {searched}"
    )


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


def pose_chain_sequences(pose):
    """Sequences of the pose, one per chain, in pose order. Used to build the
    input of a structure predictor, so that a complex is folded as a complex
    rather than as one fused polypeptide."""
    return [chain.sequence() for chain in pose.split_by_chain()]


def drain_additional_output(mover):
    """Collects every pose a one-to-many mover produced beyond the one
    already sitting in the pose apply() was given.

    Mover::get_additional_output() returns a single pose per call and None
    once there are none left, so it has to be called in a loop rather than
    wrapped in list(...)."""
    poses = []
    while True:
        extra_pose = mover.get_additional_output()
        if extra_pose is None:
            break
        poses.append(extra_pose)
    return poses


# Short names accepted by the atoms="..." attribute of an <RMSD> tag, mapped
# onto the core::scoring::rmsd_atoms enum values they stand for.
RMSD_ATOM_SETS = {
    "ca": "rmsd_protein_bb_ca",                          # alpha carbons only
    "bb": "rmsd_protein_bb_heavy",                       # N, CA, C
    "bb_o": "rmsd_protein_bb_heavy_including_O",         # N, CA, C, O
    "heavy": "rmsd_all_heavy",                           # backbone + sidechains, no hydrogens
    "all": "rmsd_all",                                   # every atom, hydrogens included
    "sc": "rmsd_sc",                                     # sidechains only
    "sc_heavy": "rmsd_sc_heavy",                         # sidechains only, no hydrogens
}


def resolve_rmsd_atoms(atoms):
    """Turns the atoms="..." value of an <RMSD> tag into an rmsd_atoms enum
    value for RMSDMetric::set_rmsd_type().

    Accepts the short names in RMSD_ATOM_SETS ("ca", "heavy", ...) or a
    core::scoring::rmsd_atoms name ("rmsd_protein_bb_ca", ...) directly. The
    chosen atom set governs the superposition as well as the measurement."""
    import pyrosetta

    rmsd_atoms = pyrosetta.rosetta.core.scoring.rmsd_atoms
    requested = atoms.strip()
    enum_name = RMSD_ATOM_SETS.get(requested.lower(), requested)
    if not hasattr(rmsd_atoms, enum_name):
        raise ValueError(
            f'Unknown atoms="{atoms}". Use one of {sorted(RMSD_ATOM_SETS)} '
            f"or a core::scoring::rmsd_atoms name."
        )
    return getattr(rmsd_atoms, enum_name)


def superimpose(moved, fixed):
    """Rotation and translation placing `moved` onto `fixed`, two equal length
    arrays of coordinates, as `moved @ rotation + translation`."""
    import numpy as np

    moved_centre = moved.mean(axis=0)
    fixed_centre = fixed.mean(axis=0)
    left, _, right = np.linalg.svd((moved - moved_centre).T @ (fixed - fixed_centre))
    handedness = np.sign(np.linalg.det(left @ right))
    rotation = left @ np.diag([1.0, 1.0, handedness]) @ right
    return rotation, fixed_centre - moved_centre @ rotation


def superimposed_rmsd(moved, fixed):
    """RMSD between two equal length arrays of coordinates, superimposed."""
    import numpy as np

    rotation, translation = superimpose(moved, fixed)
    return float(np.sqrt((((moved @ rotation + translation) - fixed) ** 2).sum(axis=1).mean()))


def match_chains(predicted, reference, kinds):
    """Pairs each chain of a prediction with a chain of the reference.

    `predicted` and `reference` hold one array of CA coordinates per chain,
    chain i of one being the chain of the same name in the other. Chain i of
    the prediction may pair with chain j of the reference only when
    `kinds[i] == kinds[j]` and the two have the same length.

    Each prediction chain that may pair with reference chain 0 is tried as an
    anchor: the prediction is superimposed on reference chain 0 through that
    chain alone, its chains are paired one to one with the reference chains
    so that the summed distance between their centres is least, and the
    pairing is scored by the CA RMSD of the whole complex superimposed.

    Returns the best scoring pairing as a list whose element i is the
    reference chain paired with prediction chain i. The chains as named win
    a tie within rounding, and are returned unchanged when they cannot pair
    as named."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    count = len(reference)
    identity = list(range(count))
    allowed = np.array([
        [kinds[i] == kinds[j] and len(predicted[i]) == len(reference[j]) for j in range(count)]
        for i in range(count)
    ])
    if len(predicted) != count or not allowed.diagonal().all():
        return identity

    whole_prediction = np.concatenate(predicted)
    predicted_centres = np.array([chain.mean(axis=0) for chain in predicted])
    reference_centres = np.array([chain.mean(axis=0) for chain in reference])

    def score(order):
        return superimposed_rmsd(whole_prediction, np.concatenate([reference[j] for j in order]))

    best, best_score = identity, score(identity)
    for anchor in np.flatnonzero(allowed[:, 0]):
        rotation, translation = superimpose(predicted[anchor], reference[0])
        centres = predicted_centres @ rotation + translation
        distances = np.linalg.norm(centres[:, None] - reference_centres[None], axis=2)
        _, order = linear_sum_assignment(np.where(allowed, distances, np.inf))
        order = order.tolist()
        order_score = score(order)
        if order_score < best_score and not np.isclose(order_score, best_score):
            best, best_score = order, order_score
    return best


# Directory a relative work_dir is kept under. rosetta_link_scripts sets it
# to the run's output directory; without it a relative work_dir lands in the
# directory the run started from.
output_root = None


def set_output_root(path):
    """Sets the directory relative work_dir attributes are kept under."""
    global output_root
    output_root = Path(path).absolute() if path else None


def resolve_work_dir(work_dir_):
    """Where a work_dir attribute ends up: as given when absolute, under
    output_root when one is set, and relative to the current directory
    otherwise."""
    if not work_dir_:
        return None
    path = Path(work_dir_)
    if path.is_absolute():
        return path
    if output_root is not None:
        return output_root / path
    return path.absolute()


@contextmanager
def work_dir(work_dir_=None, tracer_debug=None):
        original_cwd = os.getcwd()
        work_dir_ = resolve_work_dir(work_dir_)
        temp_dir = tempfile.TemporaryDirectory()
        run_dir = Path(temp_dir.name)
        if tracer_debug:
            tracer_debug << f"Using temporary directory: {run_dir} \n" and tracer_debug.flush()

        os.chdir(run_dir)
        try:
            yield run_dir
        finally:
            if work_dir_ is not None:
                os.makedirs(work_dir_, exist_ok=True)
                shutil.copytree(run_dir, work_dir_, dirs_exist_ok=True)

            # Cleanup TMP
            os.chdir(original_cwd)
            try:
                if temp_dir:
                    temp_dir.cleanup()
                    if tracer_debug:
                        tracer_debug << f"Cleaned up temporary directory {run_dir} \n" and tracer_debug.flush()
            except Exception:
                #tracer_debug << f"Failed to clean up {run_dir} \n" and tracer_debug.flush()
                pass


# --------------------------------------------------------------------------- #
# Alignments
# --------------------------------------------------------------------------- #


def read_a3m(text):
    """Records of an a3m, as (query_sequence, [(header, sequence), ...]).

    Accepts a file opening with a ColabFold "#<lengths>\t<counts>" cardinality
    line and one starting at the query record. Sequences keep their lowercase
    insertion characters.
    """
    records = []
    header = None
    parts = []
    for line in text.splitlines():
        line = line.rstrip()
        if header is None and (not line.strip() or line.startswith("#")):
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header = line
            parts = []
        elif header is not None:
            parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(parts)))
    if not records:
        raise ValueError("no sequences found in alignment")
    return records[0][1], records[1:]


def without_insertions(sequence):
    """An a3m sequence with its lowercase insertion columns removed, which
    leaves one character per column of the query."""
    return "".join(character for character in sequence if not character.islower())


def build_partial_a3m(chain_sequences, chain_alignments, chain_covered):
    """An a3m for one complex, pairing each chain with the alignment covering
    it and gapping every residue the alignments do not cover.

    chain_sequences  one sequence per chain, in chain order
    chain_alignments one a3m text per chain, or None for a chain folded from
                     its sequence alone
    chain_covered    per chain, the positions within that chain the alignment
                     covers, 0 based and ascending; None means every position

    Raises ValueError when an alignment's query length does not match the
    number of positions it is said to cover, which otherwise silently shifts
    every hit against the sequence.
    """
    lengths = [len(sequence) for sequence in chain_sequences]
    total = sum(lengths)
    offsets = []
    running = 0
    for length in lengths:
        offsets.append(running)
        running += length

    lines = [
        "#" + ",".join(str(length) for length in lengths)
        + "\t" + ",".join("1" for _ in lengths),
        ">" + "\t".join(chain_letter(index) for index in range(len(chain_sequences))),
        "".join(chain_sequences),
    ]

    # Each chain on its own, gapped outside its span, which is what makes
    # colabfold treat the record as a complex rather than one long chain.
    for index, sequence in enumerate(chain_sequences):
        before = "-" * offsets[index]
        after = "-" * (total - offsets[index] - lengths[index])
        lines.append(">" + chain_letter(index))
        lines.append(before + sequence + after)

    for index, alignment in enumerate(chain_alignments):
        if not alignment:
            continue
        covered = chain_covered[index]
        if covered is None:
            covered = list(range(lengths[index]))
        query, hits = read_a3m(alignment)
        query = without_insertions(query)
        if len(query) != len(covered):
            raise ValueError(
                f"the alignment for chain {chain_letter(index)} has {len(query)} "
                f"query columns but covers {len(covered)} residues of that chain"
            )
        positions = [offsets[index] + position for position in covered]
        for header, hit in hits:
            hit = without_insertions(hit)
            row = ["-"] * total
            for column, position in enumerate(positions):
                if column < len(hit):
                    row[position] = hit[column]
            lines.append(header)
            lines.append("".join(row))

    return "\n".join(lines) + "\n"


def chain_letter(index):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[index % len(letters)]


# --------------------------------------------------------------------------- #
# Structure prediction, shared by the ColabFold and ESMFold2 movers
# --------------------------------------------------------------------------- #
#
# The functions taking a mover read its replace_pose_, match_chains_,
# prefix_name_, rmsd_metrics, confidence_metrics and tracers.


def ca_coordinates(pose, resnums):
    """Alpha carbons of the residues given, as an array of rows."""
    import numpy as np

    return np.array([list(pose.residue(resnum).xyz("CA")) for resnum in resnums])


def residue_map(mapping):
    """A {resnum: resnum} dict as the std::map RMSDMetric takes."""
    import pyrosetta

    converted = pyrosetta.rosetta.std.map_unsigned_long_unsigned_long()
    for key, value in mapping.items():
        converted[key] = value
    return converted


def index_selector(resnums):
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector

    return ResidueIndexSelector(",".join(str(resnum) for resnum in resnums))


def selected_positions(pose, selector, limit=None):
    """Residues a selector picks, in pose numbering, up to limit residues when
    given. A selector of None is every residue."""
    last = pose.total_residue() if limit is None else min(pose.total_residue(), limit)
    if selector is None:
        return list(range(1, last + 1))
    picked = selector.apply(pose)
    return [resnum for resnum in range(1, last + 1) if picked[resnum]]


def folded_residues(pose, subset_selector):
    """Residues of one input pose that are folded, in pose numbering.
    subset_selector picks them; without it every residue is folded."""
    resnums = selected_positions(pose, subset_selector)
    if subset_selector is not None and not resnums:
        raise RuntimeError("subset_selector picks no residue, so there is nothing to fold")
    return resnums


def chains_of(pose, resnums):
    """The folded residues grouped into chains, as (chain letter, sequence,
    resnums), keeping pose order."""
    pdb_info = pose.pdb_info()
    groups = []
    for resnum in resnums:
        chain = pdb_info.chain(resnum) if pdb_info is not None else "A"
        if not groups or groups[-1][0] != chain:
            groups.append((chain, [], []))
        groups[-1][1].append(pose.residue(resnum).name1())
        groups[-1][2].append(resnum)
    return [(chain, "".join(letters), numbers) for chain, letters, numbers in groups]


def pose_records(poses, subset_selector):
    """What a predictor folds for each input pose: the records, as
    (record_id, [chain sequence, ...]), and per record the (pose, chains,
    folded resnums) its prediction is labelled from, chains as chains_of
    returns them."""
    records = []
    per_record = []
    for index, pose in enumerate(poses):
        resnums = folded_residues(pose, subset_selector)
        chains = chains_of(pose, resnums)
        per_record.append((pose, chains, resnums))
        records.append((f"design_{index}", [sequence for _, sequence, _ in chains]))
    return records, per_record


def chains_in_order(pose, resnums):
    """[(chain letter, [resnum, ...]), ...] for the residues given, chains
    in the order they first appear."""
    pdb_info = pose.pdb_info()
    chains = []
    index = {}
    for resnum in resnums:
        letter = pdb_info.chain(resnum) if pdb_info is not None else str(pose.chain(resnum))
        if letter not in index:
            index[letter] = len(chains)
            chains.append((letter, []))
        chains[index[letter]][1].append(resnum)
    return chains


def matched_chain_mapping(prediction, input_pose, folded_resnums, tracer_info):
    """The {prediction resnum: pose resnum} map that fits best, or None
    when the chains are already paired as named.

    A predictor names the chains of a homo-oligomer in whatever order it
    likes, and residue i of chain A is then compared against a different
    chain of the pose. Chains pair only with chains of the same sequence
    and length; see match_chains."""
    positions = list(range(1, prediction.total_residue() + 1))
    predicted = chains_in_order(prediction, positions)
    folded = chains_in_order(
        input_pose, [folded_resnums[position - 1] for position in positions]
    )
    if len(predicted) != len(folded) or len(predicted) < 2:
        return None

    order = match_chains(
        [ca_coordinates(prediction, members) for _, members in predicted],
        [ca_coordinates(input_pose, members) for _, members in folded],
        ["".join(input_pose.residue(resnum).name1() for resnum in members)
         for _, members in folded],
    )
    if order == list(range(len(folded))):
        return None

    mapping = {}
    for i, j in enumerate(order):
        mapping.update(zip(predicted[i][1], folded[j][1]))

    def score(pairs):
        pairs = sorted(pairs.items())
        return superimposed_rmsd(
            ca_coordinates(prediction, [position for position, _ in pairs]),
            ca_coordinates(input_pose, [resnum for _, resnum in pairs]),
        )

    as_named = dict(zip(positions, [folded_resnums[p - 1] for p in positions]))
    tracer_info << (
        f"Chains of the prediction match the pose in another order: "
        f"CA RMSD {score(as_named):.2f} as named, {score(mapping):.2f} matched. "
        f"Measuring every RMSD on the matched chains\n"
    ) and tracer_info.flush()
    return mapping


def label_prediction(mover, prediction, record_id, scores, score_keys, input_pose, folded_resnums):
    """Carries over the reslabels of the residues a prediction was folded
    from, attaches its confidence scores and computes any configured RMSD
    metrics. Returns the pose to emit: the prediction, or a copy of the input
    pose when replace_pose is false.

    scores is the confidence json of the prediction, None when there is
    none, and score_keys the values in it reported as their mean.

    input_pose is None for a structure folded from a fasta record. Those
    have no correspondence to any pose, so no labels are carried and no
    RMSD is computed; the id is recorded as the pose comment fasta_id.

    The metrics always read the prediction, whatever replace_pose emits."""
    import pyrosetta

    if input_pose is None:
        pyrosetta.rosetta.core.pose.add_comment(prediction, "fasta_id", record_id)
        attach_prediction_scores(mover, prediction, prediction, record_id, scores, score_keys)
        if mover.rmsd_metrics:
            mover.tracer_warning << (
                "Skipping RMSD metrics: structures folded from a fasta have no "
                "correspondence to the input pose\n"
            ) and mover.tracer_warning.flush()
        return prediction

    # Residue i of the prediction is the i-th folded residue of the pose
    # it came from, which is every residue unless subset_selector picks.
    pdb_info_old = input_pose.pdb_info()
    pdb_info_new = prediction.pdb_info()
    if pdb_info_old is not None and pdb_info_new is not None:
        for position, resnum in enumerate(folded_resnums[: prediction.total_residue()], 1):
            for reslabel in pdb_info_old.get_reslabels(resnum):
                pdb_info_new.add_reslabel(position, reslabel)

    emitted = prediction if mover.replace_pose_ else input_pose.clone()

    attach_prediction_scores(mover, emitted, prediction, record_id, scores, score_keys)
    attach_rmsd_metrics(mover, emitted, prediction, input_pose, folded_resnums)
    return emitted


def attach_prediction_scores(mover, emitted, prediction, record_id, scores, score_keys):
    """Attaches the mean of each of score_keys found in scores, then every
    configured metric subtag. Nothing when scores is None."""
    import numpy as np
    from pyrosetta.rosetta.core.pose import setPoseExtraScore

    if scores is None:
        return
    for key in score_keys:
        if scores.get(key) is not None:
            value = float(np.mean(scores[key]))
            score_name = f"{mover.prefix_name_}{key}"
            setPoseExtraScore(emitted, score_name, value)
            mover.tracer_info << f"\t{record_id} {score_name}: {value}\n" and mover.tracer_info.flush()

    attach_confidence_metrics(mover, emitted, prediction, record_id, scores)


def attach_confidence_metrics(mover, emitted, prediction, record_id, scores):
    """Computes every configured metric subtag: one confidence value
    averaged over the residues its selector picks.

    A metric whose selector picks nothing is skipped and warned about,
    rather than reported as the mean of an empty selection."""
    import numpy as np
    from pyrosetta.rosetta.core.pose import setPoseExtraScore

    for metric in mover.confidence_metrics:
        quantity = metric["quantity"]
        if scores.get(quantity) is None:
            mover.tracer_warning << (
                f"Skipping metric {metric['name']}: the scores of {record_id} "
                f"have no {quantity}\n"
            ) and mover.tracer_warning.flush()
            continue

        resnums = selected_positions(prediction, metric["selector"], len(scores[quantity]))
        if not resnums:
            mover.tracer_warning << (
                f"Skipping metric {metric['name']}: its selector picks no residue "
                f"of {record_id}\n"
            ) and mover.tracer_warning.flush()
            continue

        if quantity == "pae":
            # pae is a residue by residue matrix, so restricting it to a
            # selection means the block where both residues are selected.
            values = [scores["pae"][i - 1][j - 1] for i in resnums for j in resnums]
        else:
            values = [scores[quantity][resnum - 1] for resnum in resnums]

        value = float(np.mean(values))
        score_name = f"{mover.prefix_name_}{metric['name']}"
        setPoseExtraScore(emitted, score_name, value)
        mover.tracer_info << f"\t{record_id} {score_name}: {value}\n" and mover.tracer_info.flush()


def attach_rmsd_metrics(mover, emitted, prediction, input_pose, folded_resnums):
    """Computes every configured <RMSD> between the prediction and the
    residues of the pose it was folded from, and reports it on the pose
    the mover emits.

    A selector is read on the prediction and carried to the pose it came
    from through folded_resnums, so the two sides hold the same residues
    even when subset_selector folded only part of the pose.

    The chains are matched first when match_chains is set, since a
    predictor names the chains of a homo-oligomer in whatever order it
    likes."""
    import pyrosetta
    from pyrosetta.rosetta.core.pose import setPoseExtraScore

    matched = None
    if mover.match_chains_ and mover.rmsd_metrics:
        matched = matched_chain_mapping(prediction, input_pose, folded_resnums, mover.tracer_info)
    for rmsd in mover.rmsd_metrics:
        measured = selected_positions(prediction, rmsd["calculation"])
        if not measured:
            mover.tracer_warning << (
                f"Skipping RMSD {rmsd['name']}: calculation_selector picks no "
                f"residue of the prediction\n"
            ) and mover.tracer_warning.flush()
            continue

        selectors = [
            index_selector(measured),
            index_selector([folded_resnums[position - 1] for position in measured]),
        ]

        superimpose = None
        if rmsd["alignment"] is not None:
            aligned = selected_positions(prediction, rmsd["alignment"])
            if not aligned:
                mover.tracer_warning << (
                    f"Skipping RMSD {rmsd['name']}: alignment_selector picks no "
                    f"residue of the prediction\n"
                ) and mover.tracer_warning.flush()
                continue
            superimpose = (
                index_selector(aligned),
                index_selector([folded_resnums[position - 1] for position in aligned]),
            )

        rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
        rmsd_metric.set_residue_selector(selectors[0])
        rmsd_metric.set_residue_selector_reference(selectors[1])
        rmsd_metric.set_comparison_pose(input_pose)
        if superimpose is not None:
            rmsd_metric.set_residue_selector_super(superimpose[0])
            rmsd_metric.set_residue_selector_super_reference(superimpose[1])
        if matched is not None:
            if superimpose is not None:
                # A residue mapping replaces the selectors inside
                # RMSDMetric, the superposition ones included, so the two
                # cannot be given together.
                mover.tracer_warning << (
                    f"RMSD {rmsd['name']}: measuring on the chains as named, "
                    f"since alignment_selector and matched chains cannot be "
                    f"used together\n"
                ) and mover.tracer_warning.flush()
            else:
                rmsd_metric.set_residue_mapping(
                    residue_map({position: matched[position] for position in measured})
                )
        # RMSDMetric does not align by default, which for a prediction in
        # its own reference frame measures the frame offset rather than
        # any structural difference.
        rmsd_metric.set_run_superimpose(True)
        # Governs the superposition as well as the measurement. Rosetta
        # defaults to all heavy atoms, i.e. sidechains included.
        rmsd_metric.set_rmsd_type(resolve_rmsd_atoms(rmsd["atoms"]))

        rmsd_value = rmsd_metric.calculate(prediction)
        rmsd_name = f"{mover.prefix_name_}{rmsd['name']}"
        setPoseExtraScore(emitted, rmsd_name, rmsd_value)
        mover.tracer_info << f"\t{rmsd_name}: {rmsd_value}\n" and mover.tracer_info.flush()


def selector_option(tag, option, data):
    """The residue selector an attribute names, or None when unset, which
    every caller reads as every residue."""
    from pyrosetta.rosetta.core.select.residue_selector import get_residue_selector

    if not tag.hasOption(option):
        return None
    return get_residue_selector(tag.get_option_string(option), data)


def parse_rmsd_tag(child, data):
    """One RMSD subtag.

    calculation_selector names the residues the RMSD is measured over and
    alignment_selector the residues the structures are superimposed on.
    Both name a residue selector and both default to every residue, so an
    RMSD subtag with neither measures the whole prediction."""
    return {
        "name": child.get_option_string("name"),
        "calculation": selector_option(child, "calculation_selector", data),
        "alignment": selector_option(child, "alignment_selector", data),
        "atoms": child.get_option_string("atoms") if child.hasOption("atoms") else "ca",
    }


def parse_metric_tag(child, data):
    """One metric subtag: base_metric names the per residue confidence
    value and selector the residues it is averaged over, defaulting to
    every residue."""
    base_metric = child.get_option_string("base_metric").lower()
    if base_metric not in ("plddt", "pae"):
        raise RuntimeError(
            f"metric {child.get_option_string('name')} has base_metric="
            f"{child.get_option_string('base_metric')}, which is neither pLDDT nor pAE"
        )
    return {
        "name": child.get_option_string("name"),
        "quantity": base_metric,
        "selector": selector_option(child, "selector", data),
    }


def parse_prediction_metrics(mover, tag, data):
    """Appends the RMSD and metric subtags of tag to mover.rmsd_metrics and
    mover.confidence_metrics."""
    for child in tag.getTags():
        if child.getName() == "RMSD":
            mover.rmsd_metrics.append(parse_rmsd_tag(child, data))
        elif child.getName() == "metric":
            mover.confidence_metrics.append(parse_metric_tag(child, data))
    # Reject an unknown atoms= value here rather than after the
    # prediction has already run.
    for rmsd in mover.rmsd_metrics:
        resolve_rmsd_atoms(rmsd["atoms"])
    rmsd_names = [f"{rmsd['name']} (atoms={rmsd['atoms']})" for rmsd in mover.rmsd_metrics]
    mover.tracer_info << f"RMSD metrics found: {rmsd_names}\n" and mover.tracer_info.flush()
    metric_names = [
        f"{metric['name']} ({metric['quantity']})"
        for metric in mover.confidence_metrics
    ]
    mover.tracer_info << f"Confidence metrics found: {metric_names}\n" and mover.tracer_info.flush()
