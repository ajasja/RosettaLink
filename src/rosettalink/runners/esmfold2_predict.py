# @file runners/esmfold2_predict.py
# @brief Folds one job with ESMFold2. Invoked as a subprocess by the ESMFold2
#        mover; also runnable by hand for testing.
#
# ESMFold2 (https://huggingface.co/biohub/ESMFold2) ships as a python library
# rather than a command line tool, so this script is the command line the
# mover calls:
#
#   <cmd_header> esmfold2_predict.py --job job.json
#
# cmd_header is whatever runs python in the environment that has esm and
# torch installed, e.g. "python", "conda run -n esmfold2 python", or
# "singularity run --nv esmfold2.sif python". This script imports nothing
# from rosettalink, so that environment needs only esm.
#
# job.json, written by the mover:
#   {"model": "biohub/ESMFold2", "device": "cuda",
#    "num_loops": 3, "num_sampling_steps": 50, "num_diffusion_samples": 1,
#    "seed": null, "fold_kwargs": {},
#    "chains": [{"id": "A", "sequence": "MQIF..."}],
#    "out_dir": "/path/to/run_dir"}
#
# Written into out_dir, one pair per diffusion sample:
#   prediction_<i>.pdb or prediction_<i>.cif   predicted structure
#   confidence_<i>.json                        scalar confidence metrics
#
# The esm release history spells the model class and the structure
# serialisation differently (EsmFold2Model vs ESMFold2Model,
# result.complex.to_mmcif() vs model.result_to_pdb(result) vs result.mmcif),
# so each is tried in turn and every attempt is named if none succeed. Adjust
# the three helpers below to pin one spelling for your installation.

import argparse
import json
from pathlib import Path

MODEL_CLASS_NAMES = ("EsmFold2Model", "ESMFold2Model")

# Confidence metrics read off a result, in the order they are reported.
# Values are written out exactly as the model produces them: the ESMFold2
# model card documents pLDDT on 0-100 while its own examples print values on
# 0-1, so nothing here is rescaled to match AF2 or Boltz.
CONFIDENCE_KEYS = ("plddt", "ptm", "iptm", "pae", "pde")


def load_model(model_id, device):
    """Loads the ESMFold2 weights onto device and returns the model in eval
    mode. model_id is a huggingface repo id or a local directory."""
    try:
        import esm.models.esmfold2 as esmfold2_module
    except ImportError as error:
        raise SystemExit(
            f"Could not import esm.models.esmfold2: {error}\n"
            f"ESMFold2 needs esm 3.4.0 or newer, which requires python 3.12 or newer.\n"
            f"An older esm (3.2.x and earlier) imports fine but has no esmfold2 module.\n"
            f"In the environment that cmd_header runs python from:\n"
            f"  pip install --upgrade esm"
        )

    model_class = None
    for class_name in MODEL_CLASS_NAMES:
        model_class = getattr(esmfold2_module, class_name, None)
        if model_class is not None:
            break
    if model_class is None:
        try:
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
            model_class = ESMFold2Model
        except ImportError:
            raise SystemExit(
                f"None of {MODEL_CLASS_NAMES} found in esm.models.esmfold2, and "
                f"transformers.models.esmfold2 is not importable either. "
                f"Available names: {sorted(name for name in dir(esmfold2_module) if 'old2' in name)}"
            )

    try:
        # Newer releases take the device at load time.
        return model_class.from_pretrained(model_id, device=device).eval()
    except TypeError:
        return model_class.from_pretrained(model_id).to(device).eval()


def fold(model, job):
    """Folds every chain of the job as one complex and returns the raw esm
    result. One protein entry per chain keeps the residue numbering of the
    prediction aligned with the pose the mover was handed."""
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    prediction_input = StructurePredictionInput(
        sequences=[
            ProteinInput(id=chain["id"], sequence=chain["sequence"])
            for chain in job["chains"]
        ]
    )

    fold_kwargs = {
        "num_loops": job["num_loops"],
        "num_sampling_steps": job["num_sampling_steps"],
        "num_diffusion_samples": job["num_diffusion_samples"],
    }
    if job.get("seed") is not None:
        fold_kwargs["seed"] = job["seed"]
    # fold_kwargs from the job file win, so extra_args can override any of
    # the above without a change to the mover.
    fold_kwargs.update(job.get("fold_kwargs") or {})

    print(f"Folding {len(job['chains'])} chain(s) with {fold_kwargs}", flush=True)
    return ESMFold2InputBuilder().fold(model, prediction_input, **fold_kwargs)


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
    mmcif so the mover can hand the file straight to Rosetta."""
    attempted = []

    # Model-level converters.
    for method_name, suffix in (("result_to_pdb", ".pdb"), ("result_to_cif", ".cif")):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        attempted.append(f"model.{method_name}()")
        text = method(sample)
        if isinstance(text, str):
            return text, suffix

    # Result-level converters on the predicted complex.
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

    # Plain string attributes.
    for attribute, suffix in (("pdb", ".pdb"), ("mmcif", ".cif"), ("cif", ".cif")):
        text = getattr(sample, attribute, None)
        if text is None:
            continue
        attempted.append(f"result.{attribute}")
        if isinstance(text, str):
            return text, suffix

    raise SystemExit(
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


def parse_fold_kwarg(text):
    """Parses NAME=VALUE from --fold-kwarg. VALUE is read as json when it
    parses as json (numbers, true/false, null, lists) and as a plain string
    otherwise."""
    name, separator, value = text.partition("=")
    if not separator:
        raise SystemExit(f"--fold-kwarg expects NAME=VALUE, got: {text}")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError:
        return name, value


def main():
    parser = argparse.ArgumentParser(description="Fold one job with ESMFold2.")
    parser.add_argument("--job", required=True, help="Path to the job json written by the ESMFold2 mover")
    parser.add_argument(
        "--fold-kwarg",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Extra keyword argument for the esm fold() call. Repeatable. Overrides the job file.",
    )
    args = parser.parse_args()

    job = json.loads(Path(args.job).read_text())
    job.setdefault("fold_kwargs", {})
    job["fold_kwargs"].update(dict(parse_fold_kwarg(item) for item in args.fold_kwarg))

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(job["model"], job["device"])
    samples = as_samples(fold(model, job))
    print(f"ESMFold2 returned {len(samples)} sample(s)", flush=True)

    for index, sample in enumerate(samples):
        text, suffix = structure_text(model, sample)
        structure_path = out_dir / f"prediction_{index}{suffix}"
        structure_path.write_text(text)
        confidence_path = out_dir / f"confidence_{index}.json"
        confidence_path.write_text(json.dumps(confidence_scores(sample), indent=2))
        print(f"Wrote {structure_path} and {confidence_path}", flush=True)


if __name__ == "__main__":
    main()
