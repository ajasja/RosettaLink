# @file movers/ESMFold2.py
# @brief Rosetta mover to run ESMFold2 (structure prediction), as an
#        alternative to the ColabFold and Boltz2 movers.
#
# Imports esm in process, so esm 3.4.0 or newer (python 3.12 or newer) must
# be installed alongside PyRosetta. Weights are loaded once per process,
# keyed by model and device, and reused by every apply() call.
#
# model selects the weights: biohub/ESMFold2, biohub/ESMFold2-Fast (single
# sequence, faster), or a local directory.
#
# num_loops and num_sampling_steps trade accuracy against wall clock.
# num_diffusion_samples folds the same input more than once; every sample
# beyond the first is exposed through get_additional_output().
#
# Each chain of the pose is folded as its own protein entry (ids A, B, C...
# in pose order), so a complex is folded as a complex.
#
# Each prediction is written into work_dir as prediction_<i>.pdb (or .cif)
# with confidence_<i>.json beside it, then read back into a pose.
#
# Confidence scores (plddt, ptm, iptm, pae, pde) are reported under their own
# names on whatever scale ESMFold2 produces, not rescaled to match AF2 or
# Boltz pLDDT.
#
# Nested <RMSD> tags compare the prediction against the pose the mover was
# handed, over residues carrying the given labels, superimposed first over
# those same residues. A label matching no residue is skipped with a warning
# rather than failing the run.
#
# atoms="ca" (the default) aligns and measures on alpha carbons only; "heavy"
# or "all" include sidechains. See resolve_rmsd_atoms in rosettalink.utils.

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector
from pyrosetta.rosetta.core.pose import setPoseExtraScore

from rosettalink.decorators import register_mover
from rosettalink.utils import resolve_rmsd_atoms
from rosettalink.utils import setup_tracer

# Spelling of the model class has varied between esm releases.
MODEL_CLASS_NAMES = ("EsmFold2Model", "ESMFold2Model")

# Confidence metrics read off a result, in the order they are reported.
# Values are recorded exactly as the model produces them: the ESMFold2 model
# card documents pLDDT on 0-100 while its own examples print values on 0-1,
# so nothing here is rescaled to match AF2 or Boltz.
CONFIDENCE_KEYS = ("plddt", "ptm", "iptm", "pae", "pde")

# Loaded models, keyed by (model, device). Module level rather than per
# instance because RosettaScripts clones movers - a MultiplePoseMover hands
# each pose its own clone, and per-instance state would reload the weights
# for every design.
_MODEL_CACHE = {}


def load_model(model_id, device, tracer_info):
    """Returns the ESMFold2 model for model_id on device, loading it on first
    use and reusing it afterwards. model_id is a huggingface repo id or a
    local directory."""
    cache_key = (model_id, device)
    if cache_key in _MODEL_CACHE:
        tracer_info << f"Reusing model already loaded for {cache_key}\n" and tracer_info.flush()
        return _MODEL_CACHE[cache_key]

    # Imported here, not at module scope, so this mover still registers in an
    # environment without esm (it then fails on use, with this message).
    try:
        import esm.models.esmfold2 as esmfold2_module
    except ImportError as error:
        raise RuntimeError(
            f"Could not import esm.models.esmfold2: {error}\n"
            f"ESMFold2 needs esm 3.4.0 or newer, which requires python 3.12 or newer, "
            f"installed in the same environment as PyRosetta. An older esm (3.2.x and "
            f"earlier) imports fine but has no esmfold2 module."
        )

    model_class = None
    for class_name in MODEL_CLASS_NAMES:
        model_class = getattr(esmfold2_module, class_name, None)
        if model_class is not None:
            break
    if model_class is None:
        raise RuntimeError(
            f"None of {MODEL_CLASS_NAMES} found in esm.models.esmfold2. "
            f"Available: {sorted(name for name in dir(esmfold2_module) if 'old2' in name)}"
        )

    tracer_info << f"Loading {model_id} onto {device} (first use)\n" and tracer_info.flush()
    started = time.perf_counter()
    try:
        model = model_class.from_pretrained(model_id, device=device).eval()
    except TypeError:
        # Releases that take the device separately rather than at load time.
        model = model_class.from_pretrained(model_id).to(device).eval()
    tracer_info << (
        f"Loaded {model_id} in {time.perf_counter() - started:.1f}s\n"
    ) and tracer_info.flush()

    _MODEL_CACHE[cache_key] = model
    return model


def fold(model, chains, options):
    """Folds every chain as one complex and returns the raw esm result. One
    protein entry per chain keeps the residue numbering of the prediction
    aligned with the pose the mover was handed."""
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    prediction_input = StructurePredictionInput(
        sequences=[
            ProteinInput(id=chain["id"], sequence=chain["sequence"])
            for chain in chains
        ]
    )
    return ESMFold2InputBuilder().fold(model, prediction_input, **options)


def as_samples(result):
    """Normalises the fold() return value into one entry per diffusion
    sample. A single result becomes a one-item list."""
    if isinstance(result, (list, tuple)):
        return list(result)
    for attribute in ("samples", "results", "predictions"):
        candidate = getattr(result, attribute, None)
        if isinstance(candidate, (list, tuple)):
            return list(candidate)
    return [result]


def structure_text(model, sample):
    """Serialises one sample. Returns (text, suffix), preferring pdb over
    mmcif; Rosetta reads either."""
    attempted = []

    for method_name, suffix in (("result_to_pdb", ".pdb"), ("result_to_cif", ".cif")):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        attempted.append(f"model.{method_name}()")
        text = method(sample)
        if isinstance(text, str):
            return text, suffix

    complex_object = getattr(sample, "complex", None)
    if complex_object is not None:
        for method_name, suffix in (("to_pdb", ".pdb"), ("to_mmcif", ".cif")):
            method = getattr(complex_object, method_name, None)
            if method is None:
                continue
            attempted.append(f"result.complex.{method_name}()")
            text = method()
            if isinstance(text, str):
                return text, suffix

    for attribute, suffix in (("pdb", ".pdb"), ("mmcif", ".cif"), ("cif", ".cif")):
        text = getattr(sample, attribute, None)
        if text is None:
            continue
        attempted.append(f"result.{attribute}")
        if isinstance(text, str):
            return text, suffix

    raise RuntimeError(
        f"Could not serialise the ESMFold2 result to pdb or mmcif. Tried: "
        f"{attempted or 'nothing - the result exposes none of the known names'}. "
        f"Result attributes: {sorted(name for name in dir(sample) if not name.startswith('_'))}"
    )


def to_scalar(value):
    """Mean of a tensor or array, or the number itself. None if neither."""
    mean = getattr(value, "mean", None)
    if callable(mean):
        value = mean()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def confidence_scores(sample):
    """Scalar confidence metrics of one sample, keyed as in CONFIDENCE_KEYS.
    Per-residue arrays are reduced to their mean."""
    scores = {}
    for key in CONFIDENCE_KEYS:
        value = sample.get(key) if isinstance(sample, dict) else getattr(sample, key, None)
        if value is None:
            continue
        scalar = to_scalar(value)
        if scalar is not None:
            scores[key] = scalar
    return scores


def parse_fold_kwargs(text):
    """Parses fold_kwargs="NAME=VALUE,NAME=VALUE" into a dict passed straight
    to the esm fold() call. Values are read as json when they parse as json
    (numbers, true/false, null) and as plain strings otherwise."""
    kwargs = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, value = item.partition("=")
        if not separator:
            raise RuntimeError(f"fold_kwargs expects NAME=VALUE, got: {item}")
        try:
            kwargs[name.strip()] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[name.strip()] = value
    return kwargs


class ESMFold2(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(
        self,
        model="biohub/ESMFold2",
        device="cuda",
        num_loops="3",
        num_sampling_steps="50",
        num_diffusion_samples="1",
        seed=None,
        fold_kwargs=None,
        prefix_name="ESMFold2_",
        replace_pose="1",
        work_dir=None,
        delete_dir=None,
    ):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.rmsd_metrics = []

        self.model_ = model
        self.device_ = device
        self.num_loops_ = num_loops
        self.num_sampling_steps_ = num_sampling_steps
        self.num_diffusion_samples_ = num_diffusion_samples
        self.seed_ = seed
        self.fold_kwargs_ = fold_kwargs
        self.prefix_name_ = prefix_name
        self.replace_pose_ = replace_pose
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir
        # Populated by apply(): every sample beyond the first
        # (num_diffusion_samples > 1), exposed via the standard
        # Mover::get_additional_output() mechanism - same one-to-many pattern
        # as RFDiffusion/LigandMPNN/Boltz2.
        self.additional_poses_ = []

        self.tracer_fatal, self.tracer_error, self.tracer_warning, self.tracer_info, \
            self.tracer_debug, self.tracer_trace, *_ = setup_tracer("[ESMFold2]")

        self.tracer_info << (
            f"Initializing ESMFold2:\n"
            f"\tmodel: {self.model_}\n"
            f"\tdevice: {self.device_}\n"
            f"\tnum_loops: {self.num_loops_}\n"
            f"\tnum_sampling_steps: {self.num_sampling_steps_}\n"
            f"\tnum_diffusion_samples: {self.num_diffusion_samples_}\n"
            f"\tseed: {self.seed_}\n"
            f"\tfold_kwargs: {self.fold_kwargs_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()

    def clone(self):
        copy = ESMFold2()
        copy.model_ = self.model_
        copy.device_ = self.device_
        copy.num_loops_ = self.num_loops_
        copy.num_sampling_steps_ = self.num_sampling_steps_
        copy.num_diffusion_samples_ = self.num_diffusion_samples_
        copy.seed_ = self.seed_
        copy.fold_kwargs_ = self.fold_kwargs_
        copy.prefix_name_ = self.prefix_name_
        copy.replace_pose_ = self.replace_pose_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        copy.rmsd_metrics = self.rmsd_metrics.copy()
        ESMFold2.clones_.append(copy)
        return copy

    def apply(self, pose):
        # --- working directory --- #
        # Fresh, self-cleaning temp directory every call (or a fresh
        # subdirectory of a configured work_dir) - never stored back onto
        # self.work_dir_, so this mover instance can safely be applied more
        # than once (e.g. from a RosettaScripts MultiplePoseMover).
        if self.work_dir_ is None or self.work_dir_ == "":
            temp_dir = tempfile.TemporaryDirectory()
            run_dir = Path(temp_dir.name)
            self.tracer_info << f"No working directory specified, using temporary directory: {run_dir}\n" and self.tracer_info.flush()
        else:
            temp_dir = None
            os.makedirs(self.work_dir_, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(dir=self.work_dir_))

        # Keep a copy of the pose as handed to us: source of reslabels to
        # carry forward (the predicted structure carries none) and the
        # comparison pose for the RMSD block below.
        input_pose = pose.clone()

        # --- INPUT --- #
        # One protein entry per chain, in pose order, so a complex is folded
        # as a complex rather than as one fused chain. Chain ids are assigned
        # A, B, C... in that same order, which keeps the residue numbering of
        # the prediction aligned with the input pose.
        chains = [
            {"id": chr(ord("A") + chain_index), "sequence": chain.sequence()}
            for chain_index, chain in enumerate(pose.split_by_chain())
        ]

        options = {
            "num_loops": int(self.num_loops_),
            "num_sampling_steps": int(self.num_sampling_steps_),
            "num_diffusion_samples": int(self.num_diffusion_samples_),
        }
        if self.seed_ not in (None, ""):
            options["seed"] = int(self.seed_)
        if self.fold_kwargs_:
            # fold_kwargs wins, so any further esm option can be reached
            # without a change to this mover.
            options.update(parse_fold_kwargs(self.fold_kwargs_))

        # --- RUN ESMFOLD2 --- #
        model = load_model(self.model_, self.device_, self.tracer_info)

        self.tracer_info << (
            f"Folding {len(chains)} chain(s) with {options}\n"
        ) and self.tracer_info.flush()
        started = time.perf_counter()
        samples = as_samples(fold(model, chains, options))
        self.tracer_info << (
            f"Folded {len(chains)} chain(s) into {len(samples)} sample(s) "
            f"in {time.perf_counter() - started:.1f}s\n"
        ) and self.tracer_info.flush()

        # --- OUTPUT STRUCTURES --- #
        # Written to disk before being read back, so the raw tool output of a
        # run stays available for inspection.
        designed_poses = []
        for sample_index, sample in enumerate(samples):
            text, suffix = structure_text(model, sample)
            structure_file = run_dir / f"prediction_{sample_index}{suffix}"
            structure_file.write_text(text)

            scores = confidence_scores(sample)
            confidence_file = run_dir / f"confidence_{sample_index}.json"
            with open(confidence_file, "w") as f:
                json.dump(scores, f, indent=2)
            self.tracer_info << (
                f"Wrote {structure_file} and {confidence_file}\n"
            ) and self.tracer_info.flush()

            designed_poses.append(self._load_and_label_design(structure_file, scores, input_pose))

        # Primary output: same convention as the other movers - the pose the
        # caller already holds a reference to gets the first sample.
        pose.assign(designed_poses[0])

        # Every further sample (num_diffusion_samples > 1) is handed off
        # through Mover::get_additional_output(), the standard Rosetta
        # one-to-many mechanism, instead of being silently discarded.
        self.additional_poses_ = list(designed_poses[1:])

        # --- CLEANUP --- #
        try:
            if temp_dir:
                temp_dir.cleanup()
                self.tracer_info << f"Cleaned up temporary directory: {run_dir}\n" and self.tracer_info.flush()
            elif self.delete_dir_:
                shutil.rmtree(run_dir)
                self.tracer_info << f"Deleted run directory: {run_dir}\n" and self.tracer_info.flush()
        except Exception:
            self.tracer_error << f"Failed to clean up run directory: {run_dir}\n" and self.tracer_error.flush()

    @staticmethod
    def _pose_from_structure_file(structure_file):
        """Reads a predicted structure into a pose. pose_from_file defaults
        to the pdb reader, so mmcif is requested explicitly by extension."""
        pose = pyrosetta.Pose()
        if structure_file.suffix.lower() in (".cif", ".mmcif"):
            pyrosetta.rosetta.core.import_pose.pose_from_file(
                pose,
                str(structure_file),
                False,
                pyrosetta.rosetta.core.import_pose.FileType.CIF_file,
            )
        else:
            pyrosetta.rosetta.core.import_pose.pose_from_file(pose, str(structure_file))
        return pose

    def _load_and_label_design(self, structure_file, scores, input_pose):
        """Load one predicted structure, carry over reslabels from the input
        pose, attach its confidence scores, and compute any configured RMSD
        metrics. If replace_pose is false, the returned pose keeps the
        original coordinates instead of the predicted ones."""
        if self.replace_pose_:
            pose = self._pose_from_structure_file(structure_file)
            pdb_info_old = input_pose.pdb_info()
            pdb_info_new = pose.pdb_info()
            if pdb_info_old is not None and pdb_info_new is not None:
                for resnum in range(1, min(input_pose.total_residue(), pose.total_residue()) + 1):
                    for reslabel in pdb_info_old.get_reslabels(resnum):
                        pdb_info_new.add_reslabel(resnum, reslabel)
        else:
            pose = input_pose.clone()

        # --- CONFIDENCE SCORES --- #
        if not scores:
            self.tracer_warning << (
                f"No confidence scores found on the ESMFold2 result for {structure_file.name}\n"
            ) and self.tracer_warning.flush()
        for key, value in scores.items():
            score_name = f"{self.prefix_name_}{key}"
            setPoseExtraScore(pose, score_name, float(value))
            self.tracer_info << f"\t{score_name}: {value}\n" and self.tracer_info.flush()

        # --- RMSD METRICS --- #
        for rmsd in self.rmsd_metrics:
            # reslabel_input selects on the pose the mover was handed,
            # reslabel_prediction on the prediction.
            residue_selector_input = ResiduePDBInfoHasLabelSelector(rmsd["reslabel_input"])
            residue_selector_prediction = ResiduePDBInfoHasLabelSelector(rmsd["reslabel_prediction"])

            # reslabel_superimpose, when set, superimposes over a different
            # set of residues than the RMSD is measured over - e.g. align on
            # a fixed target chain and measure a designed binder, which
            # reports placement rather than fold alone. Defaults to the
            # measured residues.
            superimpose_label = rmsd.get("reslabel_superimpose", "")
            residue_selector_super = (
                ResiduePDBInfoHasLabelSelector(superimpose_label) if superimpose_label else None
            )

            # A label that matches no residue (e.g. motif on a design with no
            # scaffolded motif) would otherwise be an RMSD over nothing.
            # Skip it and leave the score unset rather than fail the run.
            selections = [
                (rmsd["reslabel_input"], residue_selector_input, input_pose),
                (rmsd["reslabel_prediction"], residue_selector_prediction, pose),
            ]
            if residue_selector_super is not None:
                selections.append((superimpose_label, residue_selector_super, pose))
            empty_labels = [
                label for label, selector, target in selections if sum(selector.apply(target)) == 0
            ]
            if empty_labels:
                self.tracer_warning << (
                    f"Skipping RMSD {rmsd['name']}: label(s) {empty_labels} select no residues\n"
                ) and self.tracer_warning.flush()
                continue

            rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
            rmsd_metric.set_residue_selector(residue_selector_prediction)
            rmsd_metric.set_residue_selector_reference(residue_selector_input)
            rmsd_metric.set_comparison_pose(input_pose)
            if residue_selector_super is not None:
                rmsd_metric.set_residue_selector_super(residue_selector_super)
            # RMSDMetric does not align by default, which for a prediction in
            # its own reference frame measures the frame offset rather than
            # any structural difference.
            rmsd_metric.set_run_superimpose(True)
            # Governs the superposition as well as the measurement. Rosetta
            # defaults to all heavy atoms, i.e. sidechains included.
            rmsd_metric.set_rmsd_type(resolve_rmsd_atoms(rmsd["atoms"]))

            rmsd_value = rmsd_metric.calculate(pose)
            rmsd_name = f"{self.prefix_name_}{rmsd['name']}"
            setPoseExtraScore(pose, rmsd_name, rmsd_value)
            self.tracer_info << f"\t{rmsd_name}: {rmsd_value}\n" and self.tracer_info.flush()

        return pose

    def get_additional_output(self):
        # Pull-one-at-a-time: pops and returns one additional sample per
        # call, None once exhausted.
        if not self.additional_poses_:
            return None
        return self.additional_poses_.pop(0)

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.tracer_debug << f"Parsing my tag @ ESMFold2...\n\tself: {self}\n\ttag: {tag}\n\tdata: {data}\n" and self.tracer_debug.flush()

        # optional attributes
        self.model_ = tag.get_option_string("model") if tag.hasOption("model") else "biohub/ESMFold2"
        self.device_ = tag.get_option_string("device") if tag.hasOption("device") else "cuda"
        self.num_loops_ = tag.get_option_string("num_loops") if tag.hasOption("num_loops") else "3"
        self.num_sampling_steps_ = tag.get_option_string("num_sampling_steps") if tag.hasOption("num_sampling_steps") else "50"
        self.num_diffusion_samples_ = tag.get_option_string("num_diffusion_samples") if tag.hasOption("num_diffusion_samples") else "1"
        self.seed_ = tag.get_option_string("seed") if tag.hasOption("seed") else ""
        self.fold_kwargs_ = tag.get_option_string("fold_kwargs") if tag.hasOption("fold_kwargs") else ""
        self.prefix_name_ = tag.get_option_string("prefix_name") if tag.hasOption("prefix_name") else "ESMFold2_"
        self.replace_pose_ = tag.get_option_bool("replace_pose") if tag.hasOption("replace_pose") else True
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""
        self.delete_dir_ = tag.get_option_bool("delete_dir") if tag.hasOption("delete_dir") else False

        # Reject an unparseable fold_kwargs here rather than after the model
        # has already been loaded.
        if self.fold_kwargs_:
            parse_fold_kwargs(self.fold_kwargs_)

        self.tracer_info << (
            f"Parsed options:\n"
            f"\tmodel: {self.model_}\n"
            f"\tdevice: {self.device_}\n"
            f"\tnum_loops: {self.num_loops_}\n"
            f"\tnum_sampling_steps: {self.num_sampling_steps_}\n"
            f"\tnum_diffusion_samples: {self.num_diffusion_samples_}\n"
            f"\tseed: {self.seed_}\n"
            f"\tfold_kwargs: {self.fold_kwargs_}\n"
            f"\tprefix_name: {self.prefix_name_}\n"
            f"\treplace_pose: {self.replace_pose_}\n"
            f"\twork_dir: {self.work_dir_}\n"
            f"\tdelete_dir: {self.delete_dir_}\n"
        ) and self.tracer_info.flush()

        # RMSD metrics (same nested tag as ColabFold and Boltz2)
        for child in tag.getTags():
            if child.getName() == "RMSD":
                self.rmsd_metrics.append({
                    "name": child.get_option_string("name"),
                    "reslabel_input": child.get_option_string("reslabel_input"),
                    "reslabel_prediction": child.get_option_string("reslabel_prediction"),
                    "reslabel_superimpose": (
                        child.get_option_string("reslabel_superimpose")
                        if child.hasOption("reslabel_superimpose") else ""
                    ),
                    "atoms": (
                        child.get_option_string("atoms")
                        if child.hasOption("atoms") else "ca"
                    ),
                })
        # Reject an unknown atoms= value here rather than after the
        # prediction has already run.
        for rmsd in self.rmsd_metrics:
            resolve_rmsd_atoms(rmsd["atoms"])
        rmsd_names = [f"{rmsd['name']} (atoms={rmsd['atoms']})" for rmsd in self.rmsd_metrics]
        self.tracer_info << f"RMSD metrics found: {rmsd_names}\n" and self.tracer_info.flush()

    @staticmethod
    def mover_name():
        return "ESMFold2"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_boolean

        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "model",
            XMLSchemaType(xs_string),
            "Weights to fold with: a huggingface repo id such as biohub/ESMFold2 or biohub/ESMFold2-Fast, or a local directory. Loaded once per process and reused, so folding many designs with one value costs one load",
            "biohub/ESMFold2"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "device",
            XMLSchemaType(xs_string),
            "Torch device the model is loaded onto. ESMFold2 expects a CUDA GPU; cpu is far slower",
            "cuda"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "num_loops",
            XMLSchemaType(xs_string),
            "Number of trunk loops. More is more accurate and slower",
            "3"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "num_sampling_steps",
            XMLSchemaType(xs_string),
            "Number of diffusion sampling steps. More is more accurate and slower",
            "50"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "num_diffusion_samples",
            XMLSchemaType(xs_string),
            "Number of structures to sample for the same input. Every sample beyond the first is exposed via get_additional_output()",
            "1"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "seed",
            XMLSchemaType(xs_string),
            "Random seed for sampling. Left unset (the esm default applies) unless explicitly provided",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "fold_kwargs",
            XMLSchemaType(xs_string),
            "Further keyword arguments for the esm fold() call, as NAME=VALUE pairs separated by commas, e.g. noise_scale=1.0,step_scale=1.5. Values are read as json when they parse as json. These override the attributes above",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "prefix_name",
            XMLSchemaType(xs_string),
            "Prefix to all metrics calculated by the mover",
            "ESMFold2_"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "replace_pose",
            XMLSchemaType(xs_boolean),
            "Whether the current pose is replaced with the predicted structure",
            "true"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "work_dir",
            XMLSchemaType(xs_string),
            "Parent directory under which each apply call gets its own fresh subdirectory holding the predicted structures and their confidence json (so this mover instance can safely be applied more than once, e.g. from a RosettaScripts MultiplePoseMover). If not provided, a new temporary directory is used per call instead. Warning: do not set the value of this attribute to empty string, as it will cause an error in pyrosetta.",
            ""))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "delete_dir",
            XMLSchemaType(xs_boolean),
            "Whether to delete this call run subdirectory after every sample has been read from disk into the primary pose or the additional-output poses",
            "false"))

        rmsd_attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        rmsd_attrlist.append(
            XMLSchemaAttribute.required_attribute(
                "name",
                XMLSchemaType(xs_string),
                "RMSD metric name"
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.required_attribute(
                "reslabel_input",
                XMLSchemaType(xs_string),
                "Residue label in input over which RMSD is calculated"
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.required_attribute(
                "reslabel_prediction",
                XMLSchemaType(xs_string),
                "Residue label in prediction over which RMSD is calculated"
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.attribute_w_default(
                "reslabel_superimpose",
                XMLSchemaType(xs_string),
                "Residue label to superimpose over, when it should differ from the residues the RMSD is measured over. For example superimpose on a fixed target chain and measure a designed binder, which reports how well the binder is placed and not only how well it folds. Defaults to the measured residues.",
                ""
            )
        )
        rmsd_attrlist.append(
            XMLSchemaAttribute.attribute_w_default(
                "atoms",
                XMLSchemaType(xs_string),
                "Atoms used for both the superposition and the RMSD: ca (alpha carbons only), bb (N, CA, C), bb_o (N, CA, C, O), heavy (backbone and sidechains, no hydrogens), all (every atom), sc or sc_heavy (sidechains only). A core::scoring::rmsd_atoms name is also accepted.",
                "ca"
            )
        )

        description = "Runs ESMFold2 to predict protein structures, as an alternative to ColabFold and Boltz2. Requires esm installed alongside PyRosetta."

        subelements = pyrosetta.rosetta.utility.tag.XMLSchemaSimpleSubelementList()
        subelements.add_simple_subelement("RMSD", rmsd_attrlist, "RMSD applied over specified residue label")
        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes_and_repeatable_subelements(
            xsd, cls.mover_name(), description, attrlist, subelements)


@register_mover
class ESMFold2Creator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = ESMFold2()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return ESMFold2.mover_name()

    def provide_xml_schema(self, xsd):
        ESMFold2.provide_xml_schema(xsd)
