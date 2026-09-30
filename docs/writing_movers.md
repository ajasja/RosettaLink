# Writing RosettaLink movers for external programs

Reference for writing a mover that wraps an external program, whether it is
a new capability or a replacement for one of the existing pipeline stages.
Written to be read before writing the next one.

**Part 1 (sections 1-8)** applies to any mover: the class skeleton, wrapping
a subprocess, carrying metadata across the disk boundary, one-to-many
output, composing stages in RosettaScripts, reporting metrics, RMSD, and how
to verify an unfamiliar Rosetta component cheaply.

**Part 2 (sections 9-11)** is about fitting into this pipeline: the contract
each role must satisfy for a replacement to be drop-in, what the current
movers actually do, and how to stay backwards compatible.

Read the existing movers alongside this doc - they are the worked examples
every section refers back to:
- src/rosettalink/movers/RFDiffusion.py - backbone generation
- src/rosettalink/movers/LigandMPNN.py - sequence design
- src/rosettalink/movers/ColabFold.py - structure prediction
- src/rosettalink/movers/Boltz2.py - structure prediction, multi-chain


## 1. The shape every mover has

Every RosettaLink mover follows the same skeleton. All of it is boilerplate
except apply():

```python
class MyTool(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()

    def __init__(self, some_option=None, work_dir=None, delete_dir=None):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.some_option_ = some_option
        self.work_dir_ = work_dir
        self.delete_dir_ = delete_dir
        self.additional_poses_ = []  # see section 4 - only if one-to-many

        self.tracer_fatal, self.tracer_error, self.tracer_warning, \
            self.tracer_info, self.tracer_debug, self.tracer_trace, *_ = \
            setup_tracer("[MyTool]")

    def clone(self):
        copy = MyTool()
        copy.some_option_ = self.some_option_
        copy.work_dir_ = self.work_dir_
        copy.delete_dir_ = self.delete_dir_
        MyTool.clones_.append(copy)
        return copy

    def apply(self, pose):
        ...  # section 2 and 3

    def get_name(self):
        return self.mover_name()

    def parse_my_tag(self, tag, data):
        self.some_option_ = tag.get_option_string("some_option") if tag.hasOption("some_option") else ""
        self.work_dir_ = tag.get_option_string("work_dir") if tag.hasOption("work_dir") else ""
        self.delete_dir_ = tag.get_option_bool("delete_dir")

    @staticmethod
    def mover_name():
        return "MyTool"

    @classmethod
    def provide_xml_schema(cls, xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xs_string, xs_boolean
        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "some_option", XMLSchemaType(xs_string), "...", ""))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "delete_dir", XMLSchemaType(xs_boolean), "..."))
        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd, cls.mover_name(), "Runs MyTool.", attrlist)


@register_mover
class MyToolCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)

    def create_mover(self):
        mover = MyTool()
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return MyTool.mover_name()

    def provide_xml_schema(self, xsd):
        MyTool.provide_xml_schema(xsd)
```

Then register the module so rosettalink.init() actually imports it: add
its name to REGISTRATION_MODULES['movers'] in
src/rosettalink/centralized_registration.py. Forgetting this step means
the mover silently does not exist to RosettaScripts (no error - init()
just never imports the file, so @register_mover never runs).

Schema description strings cannot contain the characters `<`, `>` or `&`.
Rosetta embeds them verbatim in the generated XSD document, so it validates
and rejects them up front:

```
Desciption for attribute "cmd_header" may not contain either '<', '>' or '&'.
```

This applies to every string passed to `XMLSchemaAttribute.*`, to
`add_simple_subelement`, and to the `description` argument - so a
placeholder in an example has to be written without angle brackets
(`conda run -n ENVNAME ...`, not `conda run -n <env> ...`). It fails at
`XmlObjects.create_from_string()`, before any mover runs, so it costs
nothing but is easy to reintroduce when editing a description.

Constructor arguments are conventionally strings even for booleans/numbers
(num_designs="2", not 2) when a mover is meant to be usable identically
from Python and from XML - parse_my_tag always hands back strings anyway
(except for attributes read with get_option_bool/get_option_int), so
keeping the Python-object constructor consistent avoids two code paths
having different type expectations for the same option.

Two log lines on construction is normal, not a bug. RosettaScripts always
constructs a mover via its Creator with no arguments first (so __init__
logs every default), then calls parse_my_tag() on that same instance right
after (which logs the real values from the XML). One mover, two lifecycle
stages, two log blocks - not two runs.


## 2. Reaching the external program

The first decision, because it shapes everything else:

- **The tool has a command line** - run it as a subprocess. Its
  dependencies live wherever it is installed and can never reach the
  PyRosetta environment, so a version conflict between two wrapped tools is
  impossible.
- **The tool is a library only** - the mover has to import it, so it has to
  be installed alongside PyRosetta. Check that it can be before writing
  anything: python version, torch build, CUDA major version. See
  `docs/mover_setup.md`.

When both are possible, the deciding question is what the tool holds
per process. A subprocess starts fresh every apply(), so anything it loads
is paid for once per design. A tool whose model weights cost more to load
than the design costs to run belongs in process, where the mover can load
them once and reuse them; a tool that reloads them anyway, like a CLI, is
free to live wherever is convenient.

### As a subprocess

The pattern (singularity here, but the same shape fits any subprocess-based
tool): dump the pose, run the container bind-mounted to a working
directory, read whatever files it wrote back into pose objects.

```python
run_and_log(cmd_str, self.tracer_info, self.tracer_error)
```

run_and_log (in rosettalink/utils.py) raises if the subprocess exits
non-zero - treat a failed external run as fatal, do not try to limp on with
partial output.

### The working-directory bug that bit every one of these movers

Never write the resolved directory back onto self.work_dir_. The original
code for all three movers did this:

```python
# WRONG - do not do this
if self.work_dir_ is None or self.work_dir_ == "":
    temp_dir = tempfile.TemporaryDirectory()
    self.work_dir_ = temp_dir.name   # <-- mutates the mover own config
```

This looks harmless and works fine the first time. It breaks the moment the
same mover instance is applied a second time - which native
MultiplePoseMover does routinely (it parses one mover instance per stage
and reapplies it once per pose it collects; see section 5). The second call
sees self.work_dir_ already set to the first call directory, reuses it,
and silently overwrites or collides with the first call files.

The fix - always resolve into a local variable, never touch
self.work_dir_:

```python
def apply(self, pose):
    if self.work_dir_ is None or self.work_dir_ == "":
        temp_dir = tempfile.TemporaryDirectory()
        run_dir = Path(temp_dir.name)
    else:
        # A configured work_dir can ALSO be reused across many apply() calls
        # on this instance - each call gets its own fresh subdirectory under
        # it instead of writing directly into it.
        temp_dir = None
        os.makedirs(self.work_dir_, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(dir=self.work_dir_))

    # ... use run_dir everywhere below, never self.work_dir_ ...

    try:
        if temp_dir:
            temp_dir.cleanup()
        elif self.delete_dir_:
            shutil.rmtree(run_dir)   # only this call own subdirectory
    except Exception:
        self.tracer_debug << f"Failed to clean up {run_dir}\n" and self.tracer_debug.flush()
```

Test this specifically by calling apply() on the same mover object twice in
a row before considering a mover done - a Python loop that builds a fresh
mover object every iteration will never catch this bug.

### Picking out real output files, not just globbing star-dot-pdb

If the mover also writes its own input pdb into the same directory the
external tool writes its output into, a bare glob('*.pdb') will pick that
up too. Do not rely on filename or sort-order to exclude it (RFDiffusion
input.pdb happened to sort after its own numbered outputs, which is a
coincidence, not a rule). Instead key off whatever the tool own output
convention actually is - e.g. RFDiffusion only writes a .trb file next to
genuine designs:

```python
design_pdb_files = [f for f in pdb_files if f.with_suffix('.trb').is_file()]
```

### Quote the values you interpolate

run_and_log goes through os.system, so the command string is parsed by
/bin/sh. An option value containing a pipe, a semicolon or a space is shell
syntax unless it is quoted:

```python
cmd_str += f' --{name} "{value}"'
```

The same applies in the other direction: a cmd_header set from an XML
attribute cannot contain a bare `&`, which XML forbids. To put environment
variables in front of a command, use the assignment prefix form
(`VAR=value cmd`), which is one command and needs no separator at all.

### Exposing a large command line

A tool with many options does not need an attribute block each. Keep one
dict of name to description and let it drive both parsing and the schema, so
adding an option is a single line:

```python
OPTION_HELP = {
    "n_res": "Size of the desired network, in residues. Tool default is 2",
    # one entry per option
}

# parse_my_tag
self.options_ = {
    name: (tag.get_option_string(name) if tag.hasOption(name) else "")
    for name in OPTION_HELP
}

# provide_xml_schema
for name, help_text in OPTION_HELP.items():
    attrlist.append(XMLSchemaAttribute.attribute_w_default(
        name, XMLSchemaType(xs_string), f"{help_text}. Passed through only when set", ""))
```

Passing each option through only when it is set leaves the tool own defaults
in force, so the mover cannot silently disagree with them.

### In process

A library with no command line has to be imported. Three things follow, none
of them about any particular tool:

**Cache expensive state at module level, keyed by config.** RosettaScripts
clones movers - MultiplePoseMover hands each pose its own clone - so state on
the instance gives you one load per pose, which is the cost going in process
was meant to avoid.

```python
_MODEL_CACHE = {}

def load_model(model_id, device, tracer_info):
    cache_key = (model_id, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    ...
    _MODEL_CACHE[cache_key] = model
    return model
```

**Import inside the method, not at module scope.** centralized_registration
catches ImportError and reports the module as optional, so a top-level import
of a missing dependency makes the mover disappear from the schema instead of
failing. Imported at the point of use, it raises with a message naming what
is missing.

**Without cmd_header, the environment is the interface.** There is no
attribute to point somewhere else, so where the mover expects to be installed
is part of its documented behaviour and belongs in the file header.


## 3. Carrying metadata across a dump-to-disk-and-reload boundary

Loading the external tool output PDB with pyrosetta.pose_from_file() gives
a brand-new Pose with none of the input pose pdb_info, reslabels, or extra
scores - all of that lives in Rosetta in-memory pose cache, not in the PDB
file the external tool wrote. Anything that should survive the round trip
has to be copied over by hand:

```python
pdb_info_old = input_pose.pdb_info()
pdb_info_new = new_pose.pdb_info()
for resnum in range(1, min(input_pose.total_residue(), new_pose.total_residue()) + 1):
    for reslabel in pdb_info_old.get_reslabels(resnum):
        pdb_info_new.add_reslabel(resnum, reslabel)
```

(LigandMPNN._load_and_label_design and the "REPLACE POSE" block in
ColabFold both do exactly this.)

Always also stamp a label spanning every residue (RosettaLink convention is
"all"), in addition to whatever conditional/subset labels the tool own
output implies. A conditional label can legitimately be empty - e.g.
RFDiffusion inpaint_seq/inpaint_str labels mark residues kept from an
input, which is nothing at all for a fully unconditional design - and
downstream consumers (e.g. an RMSD metric that needs the whole chain) have
no fallback if the only label available to select on selects zero
residues.

A ResidueIndexSelector cannot take an empty string. If the resnum list
being built can legitimately be empty, guard it:

```python
selector = ResidueIndexSelector(resnums) if resnums else FalseResidueSelector()
```

A label that matches no residue should not be applied, and should not be
logged either - a line reading "Labelled 0 residue(s) as motif" is
indistinguishable from a measured result of zero. Skip both.

For the labels the current movers actually use, see section 10.

## 4. One-to-many movers: get_additional_output()

If a tool can produce more than one result per call (RFDiffusion
num_designs, LigandMPNN batch_size), do not just keep the first result and
discard the rest. Implement get_additional_output() - the standard Rosetta
mechanism for "one apply(), many poses" - so every consumer (plain Python,
RosettaScripts MultiplePoseMover, JD2) picks up every result the same way,
without any RosettaLink-specific handling on their end.

The mechanism carries no ordering contract: a caller cannot tell whether the
poses are interchangeable samples or a ranked list. If the tool ranks its
output, say so in the file header, and put the best one in the pose.

It is a pull-one-at-a-time API, not a bulk fetch. This was verified
empirically, not assumed: calling get_additional_output() on a plain
Rosetta mover that does not override it (MutateResidue) returns Python
None - that is the genuine C++ default, "nothing more to give you", not an
empty list or vector. So each call must pop and return exactly one pose,
returning None once exhausted:

```python
def apply(self, pose):
    designs = [... every result from this call, including the primary one ...]
    pose.assign(designs[0])
    self.additional_poses_ = list(designs[1:])   # reset fresh every apply()

def get_additional_output(self):
    if not self.additional_poses_:
        return None
    return self.additional_poses_.pop(0)
```

Returning the whole list in one call looks like it works if the only
caller is a Python driver script (list(mover.get_additional_output()) just
copies a list, no casting involved) - it fails loudly the moment real C++
Rosetta code (MultiplePoseMover) calls it, because pybind11 then has to
cast whatever Python object comes back into the C++ return type the caller
expects, and a plain list does not survive that cast (RuntimeError: Unable
to cast Python instance of type list to C++ type). Test this specifically
by driving the mover through a MultiplePoseMover before considering it
done - pure-Python callers will not catch this either.

To collect everything from a one-to-many mover in Python, use
rosettalink.utils.drain_additional_output(mover), which loops until None:

```python
from rosettalink.utils import drain_additional_output
backbones = [seed_pose] + drain_additional_output(rfdiff_mover)
```


## 5. Composing one-to-many movers natively in RosettaScripts

MultiplePoseMover is the native Rosetta mechanism for chaining one-to-many
stages inside a single PROTOCOLS list, without any Python loop at all (per
the RosettaScripts wiki page for it):

- It collects every pose from the mover immediately before it in the same
  PROTOCOLS list (the primary pose plus everything from that mover
  get_additional_output()).
- It runs its own nested ROSETTASCRIPTS sub-protocol on each one.
- Its own output (one plus additional, same convention) can feed into
  another mover, so MultiplePoseMover instances chain back-to-back, and a
  sub-protocol can itself contain one or more MultiplePoseMover instances -
  which is how a two-level fan-out (N backbones then M sequences each then
  a prediction each) becomes one XML document:

```xml
<MOVERS>
    <RFDiffusion name="make_backbone" num_designs="2" .../>

    <MultiplePoseMover name="design_sequences">
        <ROSETTASCRIPTS>
            <MOVERS>
                <LigandMPNN name="make_sequence" batch_size="3" .../>
                <MultiplePoseMover name="predict_structures">
                    <ROSETTASCRIPTS>
                        <MOVERS><ColabFold name="predict_structure" .../></MOVERS>
                        <PROTOCOLS><Add mover="predict_structure" /></PROTOCOLS>
                    </ROSETTASCRIPTS>
                </MultiplePoseMover>
            </MOVERS>
            <PROTOCOLS>
                <Add mover="make_sequence" />
                <Add mover="predict_structures" />
            </PROTOCOLS>
        </ROSETTASCRIPTS>
    </MultiplePoseMover>
</MOVERS>
<PROTOCOLS>
    <Add mover="make_backbone" />
    <Add mover="design_sequences" />
</PROTOCOLS>
```

Driving it from Python is then one call plus one drain, on the outer
ParsedProtocol:

```python
protocol = XmlObjects.create_from_string(xml_string).get_mover("ParsedProtocol")
protocol.apply(seed_pose)
all_final_poses = [seed_pose] + drain_additional_output(protocol)
```

This only works if every mover in the chain both (a) correctly implements
get_additional_output() (section 4) and (b) never mutates its own
work_dir_ between calls (section 2) - MultiplePoseMover reapplies the same
mover instance once per pose it collects, it does not construct a fresh
one per branch. Confirm both before wiring a mover into a MultiplePoseMover
chain.


## 6. Reporting metrics and filters for later analysis (e.g. a CSV)

confidence="0" on a Filter added to PROTOCOLS means "compute and cache the
value, never reject the trajectory on it" - ParsedProtocol caches it into
the pose extra scores under the filter own name, with no need to go
through JD2 job distributor. Confirmed directly in a real run log:
protocols.rosetta_scripts.ParsedProtocol: Set filter value for netcharge
to: -3.

SimpleMetrics (the more modern equivalent) do not run themselves just by
being declared in SIMPLE_METRICS - they need a RunSimpleMetrics mover
(metrics="metric_name") added to PROTOCOLS to actually execute and cache
them the same way a confidence=0 filter does.

Either way the values end up in pose.scores (a dict-like accessor) by the
time apply()/get_additional_output() hand back the final poses. Since a
plain Python driver script is not going through JD2, there is no automatic
score-file or CSV writer - read pose.scores and write the CSV directly
(see examples/demo_full_pipeline_single_xml_extended.py).


## 7. Comparing a prediction against its input with RMSDMetric

A structure-prediction mover typically wants to report how far the
prediction moved from the pose it was handed. Both ColabFold and Boltz2 do
this through nested `<RMSD>` tags:

```xml
<RMSD name="scRMSD_designed"          reslabel_input="designed" reslabel_prediction="designed" />
<RMSD name="scRMSD_binder_on_target"  reslabel_input="designed" reslabel_prediction="designed"
                                      reslabel_superimpose="fixed_chain" />
<RMSD name="scRMSD_designed_sidechains" reslabel_input="designed" reslabel_prediction="designed"
                                      atoms="heavy" />
```

Five things about `RMSDMetric` that are easy to get wrong:

**It does not superimpose by default.** From its own docstring: "Please note
that by default it doesn't align poses. You need to set
set_run_superimpose() to True for performing alignment." An external tool
returns coordinates in its own reference frame, so without this the metric
measures the frame offset, not any structural difference. The symptom is
plausible-looking but far too large values - tens of Angstroms, roughly
constant across designs, on structures that align to under 2 A by hand.

**It measures every heavy atom by default, sidechains included.**
`rmsd_type` defaults to `rmsd_all_heavy`. A self-consistency RMSD is
conventionally over alpha carbons, so set it:

```python
rmsd_metric.set_rmsd_type(pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca)
```

The atom set governs the superposition as well as the measurement - inside
`RMSDMetric::calculate` the same `rmsd_type_` is applied to the metric that
builds the superposition atom map - so this gives a CA-on-CA alignment and a
CA RMSD. Leaving the default in compares rotamers as well as fold and reads
noticeably higher than the number another pipeline reports for the same pair
of structures. The full enum is `rmsd_protein_bb_ca`,
`rmsd_protein_bb_heavy`, `rmsd_protein_bb_heavy_including_O`, `rmsd_sc`,
`rmsd_sc_heavy`, `rmsd_all_heavy`, `rmsd_all`; `rosettalink.utils`
`resolve_rmsd_atoms()` maps the short `atoms=` values onto them.

**Superposition uses the measured residues unless told otherwise.**
`set_residue_selector_super` (and optionally
`set_residue_selector_super_reference`, which falls back to the former)
takes a different selection to align on. This is the difference between two
distinct questions:

- align and measure the same subset - does that part fold as designed
- align on a fixed chain, measure a designed one - is the designed part also
  *placed* as designed

For binder design the second is the docking-accuracy number; the first
cannot see a correctly folded binder docked in the wrong place.

**The selectors are per-pose, and which is which matters.**
`set_residue_selector` applies to the pose passed to `calculate()`;
`set_residue_selector_reference` applies to the pose given to
`set_comparison_pose()`. Both must select the same number of residues.
Keeping the labels identical on both poses (section 3) makes this automatic.

**Guard empty selections.** A label matching no residue is an RMSD over
nothing. Skip the metric and leave the score unset rather than reporting a
number or failing the run:

```python
if sum(selector.apply(pose)) == 0:
    ...  # warn naming the label, then continue
```

Putting it together:

```python
rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
rmsd_metric.set_residue_selector(selector_on_prediction)
rmsd_metric.set_residue_selector_reference(selector_on_input)
rmsd_metric.set_comparison_pose(input_pose)
if selector_to_align_on is not None:
    rmsd_metric.set_residue_selector_super(selector_to_align_on)
rmsd_metric.set_run_superimpose(True)
rmsd_metric.set_rmsd_type(resolve_rmsd_atoms(atoms))   # "ca" by default
value = rmsd_metric.calculate(pose)
```

A free sanity check on any real run: aligning on a subset is by definition
the best alignment for measuring that subset, so

```
rmsd(measure X, align on Y)  >=  rmsd(measure X, align on X)
```

must hold. A violation means the superposition selector is not being
applied. Likewise an RMSD of a fixed chain against itself should come out
near zero.


## 8. Verifying an unfamiliar Rosetta component before spending a GPU job on it

Every mistake in this file that actually shipped and had to be fixed later
was an assumption about Rosetta or pybind11 internals that turned out
wrong, made from memory when a real source of truth was available and
should have been checked first. In order of preference:

**First choice: fetch the real docs.** `docs.rosettacommons.org` is public,
not paywalled, and has a page per mover/filter/simple metric with the real
XML syntax, attributes, defaults, and semantics - e.g.
`https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/RosettaScripts-<MoverName>`
for movers (filters and simple metrics live under similar
`scripting_documentation/RosettaScripts/...` paths - search for the exact
one rather than guessing the URL). Use WebSearch to find the exact page,
then WebFetch it, before writing XML for a component that has not been
looked up this session. This is strictly better than asking whoever is
running the job to look something up and paste it back - fetch it directly
instead.

**Second choice: dump the schema from the actual installed build.** If the
docs site does not have a page for something (or the installed Rosetta
version might differ from the docs site's "latest"), ask whoever has
cluster access to run:
```python
import pyrosetta
pyrosetta.init("-parser:info SomeMoverOrFilterName")
```
This works because PyRosetta is a full Rosetta build - no separate
rosetta_scripts binary needed. It has the advantage of matching the exact
installed version, at the cost of a round trip through a person instead of
a direct fetch.

**Third: discover what a virtual method genuinely returns, rather than
guessing.** Construct an unrelated, already-known-concrete mover that does
not override the method in question, call it, and inspect the real type or
value that comes back - this is how the get_additional_output() pull-one-
at-a-time contract in section 4 was confirmed, and it is not something
either docs site or -parser:info documents directly:
```python
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
m = MutateResidue()
print(type(m.get_additional_output()))   # None - the genuine C++ default
```

**Fourth: replay the logic offline against artifacts from a previous run.**
Index arithmetic and residue bookkeeping can be checked without PyRosetta
and without a GPU, because a finished run leaves behind everything needed.
The `.trb` needs only pickle and numpy, and an output `.pdb` gives the
residue order Rosetta would load, so `pose.pdb_info().pdb2pose()` can be
simulated with a dict built from the ATOM records:

```python
order, seen = [], set()
for line in open("_0.pdb"):
    if line.startswith("ATOM"):
        key = (line[21], int(line[22:26]))          # chain, resnum
        if key not in seen:
            seen.add(key); order.append(key)
pdb2pose = {k: i + 1 for i, k in enumerate(order)}  # 1-based pose numbering
```

Compute the old and the new logic side by side over that and compare the
counts against what the run actually logged. This settles an off-by-frame
or wrong-numbering bug in seconds and confirms a fix before committing a
GPU job to it, which a code read alone cannot.

**Fifth: stub the wrapped tool.** The mover can be exercised end to end
before the tool is installed, and before any GPU time, by putting a fake
version of it on the import path or on PATH. Have the stub emit the file
layout the mover expects - the output structures, the score file, the
directory nesting - and run apply() against it. This exercises the whole
path the tool does not participate in: command construction, output
discovery, pose loading, label carry-forward, score parsing, RMSD.

Write more than one stub. Varying what the stub does covers the branches
that only appear against a particular version of the tool - a different
class or method name, a missing optional output, several results instead of
one - and confirms that the error messages name what was actually tried.

### Cheap checks before running anything

These catch the failures that are invisible in a code read and expensive to
find in a job log:

- **Attribute sets agree** across `__init__`, `clone()` and
  `parse_my_tag()`. An attribute missing from `clone()` is silently dropped
  the moment RosettaScripts clones the mover, so it works from Python and
  loses its value in a protocol.
- **No `<`, `>` or `&` anywhere in a schema description string.** Rosetta
  builds its XSD as XML and rejects them, at registration time, for every
  protocol - not just ones using this mover.
- **Every `{placeholder}` in a demo's XML template has a matching
  `.format()` keyword**, and vice versa.
- **The filled XML parses** as XML, which catches an unbalanced tag after a
  rename.

Parsing an XML string is free; running it is not. A wrong attribute name
fails immediately at XmlObjects.create_from_string(...), before any
singularity or GPU work happens - so a quick parse-only smoke test (no
.apply()) catches schema mistakes at zero cost. A wrong semantic
assumption baked into a valid schema (e.g. BuriedUnsatHbonds2 jump_number
defaulting to 1, meant for interface analysis across a jump that a
single-chain pose does not have) will NOT be caught this way - it only
surfaces at runtime, mid-job, and none of the three verification methods
above are guaranteed to surface a semantic mismatch like that either. Flag
those explicitly as unverified rather than presenting them as certain, and
expect to iterate once against a real run log.


# Part 2 - pipeline roles, the current movers, and compatibility

Everything above applies to any mover. This part covers what a mover has to
provide to stand in for one of the existing pipeline stages.


## 9. What each pipeline role must provide

The pipeline works on shared conventions, not on the specific tools. A
replacement for any stage is drop-in as long as it honours the contract for
that role. Where the current movers already satisfy a point, it is listed
because a replacement must too.

These roles are positions in the pipeline. A mover wrapping a tool that
occupies no position has no contract to satisfy beyond the mover skeleton in
Part 1 - there is nothing further to look up here.

### Backbone generation (the RFDiffusion role)

- Number of designs is a mover attribute, not repeated apply() calls. First
  design goes into the pose that was handed in, the rest through
  get_additional_output() (section 4).
- If the tool needs an input structure, dump the incoming pose into the run
  directory and point the tool at that path. Callers pass the real starting
  structure as the pose, so it must not be treated as a placeholder.
- Stamp `all` on every residue. Downstream selectors need one label that is
  never empty.
- If the tool distinguishes residue provenance, stamp `fixed_chain`, `motif`
  and `designed` as an exact non-overlapping partition, plus `inpainted`
  where the tool can keep a backbone while redesigning identity. Omit any
  label that matches nothing rather than applying it empty.
- Store the `inpaint_seq` and `inpaint_str` pose-cache subsets if the tool
  has equivalents, so StoredResidueSubset in XML keeps working.

### Sequence design (the LigandMPNN role)

- Do not change the residue count, the residue order, or the backbone
  coordinates. Every downstream label index and RMSD residue correspondence
  assumes a 1:1 mapping with the pose that came in. A tool that renumbers or
  reorders residues breaks the labels silently.
- Carry every reslabel forward from the input pose onto each output pose.
  The tool own output PDB carries none (section 3).
- Number of sequences is a mover attribute. First sequence into the pose,
  the rest through get_additional_output().

### Structure prediction (the ColabFold / Boltz2 role)

- Accept a multi-chain pose and fold it as a complex - one input entry per
  chain, in pose order. Concatenating `pose.sequence()` into a single chain
  silently folds a complex as one fused polypeptide and makes every
  downstream number meaningless.
- Capture `input_pose = pose.clone()` before overwriting the pose. It is
  both the source of reslabels to carry forward and the comparison pose for
  RMSD.
- Honour `replace_pose`, and prefix every score written with `prefix_name`.
- Support nested `<RMSD>` tags with `reslabel_input`,
  `reslabel_prediction`, `reslabel_superimpose` and `atoms` (section 7).
  `atoms` defaults to `ca`, not to Rosetta own all-heavy default, so that
  two prediction movers report comparable numbers.


## 10. The current movers

Read these alongside their source; the details below are the parts not
obvious from the code.

| mover | role | reached as | fan-out attribute | score prefix convention |
|---|---|---|---|---|
| RFDiffusion | backbone generation | subprocess, `rfdiffusion_path` | `num_designs` | n/a, writes labels not scores |
| LigandMPNN | sequence design | subprocess, `ligandmpnn_path` | `batch_size` (x `number_of_batches`) | n/a |
| ColabFold | structure prediction | subprocess, `cmd_header` | none (one model per call) | `prefix_name`, default `AF2_` |
| Boltz2 | structure prediction | subprocess, `cmd_header` | `diffusion_samples` | `prefix_name`, default `Boltz_` |
| ESMFold2 | structure prediction | in process | `num_diffusion_samples` | `prefix_name`, default `ESMFold2_` |
| HBDesigner | none (fixed backbone redesign) | subprocess, `cmd_header` | `top_k`, ranked best first | `prefix_name`, default `HBDesigner_` |

### RFDiffusion: the labels it stamps

`RFDiffusion._load_and_label_design` puts these on every design, and
everything downstream selects on them via
`ResiduePDBInfoHasLabelSelector`. Any label matching nothing is omitted:

| label | meaning |
|---|---|
| `all` | every residue; the only label guaranteed non-empty |
| `inpaint_str` | backbone taken from the input |
| `inpaint_seq` | identity taken from the input |
| `fixed_chain` | kept backbone outside a scaffolded motif, i.e. chains carried over whole (receptor/target) |
| `motif` | kept backbone that came from the reference as a scaffolded motif |
| `designed` | built de novo |
| `inpainted` | backbone kept but identity redesigned (`contigmap.inpaint_seq`) |

`fixed_chain`, `motif` and `designed` are an exact partition of the pose
with no overlap. `inpainted` is additive on top - those residues stay in
`fixed_chain`/`motif` so that a chain-level RMSD still covers whole chains.
Note `inpainted` is the near-inverse of `inpaint_seq`, which marks
identities that were KEPT; the names are one letter apart and mean opposite
things.

### RFDiffusion: reading .trb indices

The `.trb` sidecar is a pickle of numpy arrays. Inspect one directly when
adding or debugging labels - it needs only pickle and numpy, no PyRosetta:

```python
import pickle, numpy as np
d = pickle.load(open("_0.trb", "rb"))
print(sorted(d))
```

`inpaint_str` and `inpaint_seq` are per-residue boolean arrays, 0-based and
indexed over the output pose, so `index + 1` is the Rosetta residue number.

**The keys are not all in the same numbering, and the names do not say so.**

| key | numbering |
|---|---|
| `con_hal_idx0` | 0-based over the **output pose** - the motif residues. Empty when there is no motif |
| `complex_con_hal_idx0` | 0-based over the output pose - all kept residues |
| `receptor_con_hal_idx0` | the receptor enumerated **0..N-1 in its own numbering** - NOT a position in the output pose |
| `receptor_con_hal_pdb_idx` | `(chain, resnum)` in output **PDB** numbering; convert with `pose.pdb_info().pdb2pose(chain, resnum)` |
| `receptor_con_ref_pdb_idx` | `(chain, resnum)` in the **input/reference** PDB numbering |

Using `receptor_con_hal_idx0` as an output-pose index silently mislabels
everything whenever the receptor is not the first chain, and produces
overlapping labels rather than an error. Derive `fixed_chain` from
`inpaint_str` minus `con_hal_idx0` instead.

**Chain order in the output is not the contig order.** For the contig
`[B1-150/0 90-120]` the output pose is chain A = the 114-residue de novo
binder first, chain B = the 150-residue target second. Never assume the
receptor is chain 1 or occupies the first residues.

### LigandMPNN

Total sequences per input pose is `batch_size` x `number_of_batches`. Its
output PDBs live in a `backbones/` subdirectory of the run directory and
carry no pdb_info, so reslabels are copied forward from the input pose.

### ColabFold, Boltz2 and ESMFold2

All three fill the same role and are swappable by changing the tag name and
the tool-specific attributes. What differs:

- ColabFold takes a FASTA and reports `plddt` (0-100, averaged per-residue)
  and `pae`. It writes a single chain, so it does not currently handle a
  multi-chain pose - see the role contract in section 9.
- Boltz2 takes a YAML with one entry per chain and reports Boltz native
  0-1 scores under their own names: `confidence_score`, `ptm`, `iptm`,
  `complex_plddt`, `complex_iplddt`, `complex_pde`, `complex_ipde`.
- Boltz2 needs `use_msa_server="false"` plus the `msa: empty` sentinel it
  writes for single-sequence mode, or a network route to the MSA server.
- Boltz2 fans out over `diffusion_samples`; ColabFold returns one pose per
  call regardless of how many AF2 models it evaluates internally.

- ESMFold2 is the one predictor that runs in process, so it needs esm in the
  PyRosetta environment and loads its weights once per process rather than
  once per design. It takes one protein entry per chain and reports `plddt`,
  `ptm`, `iptm`, `pae` and `pde` on whatever scale esm produces, and fans
  out over `num_diffusion_samples`.

The pLDDT scales differ (0-100 versus 0-1) which is why the keys are named
differently rather than both being `plddt`. Do not unify them.

### HBDesigner

Designs buried hydrogen bond networks into a fixed backbone, so it occupies
no pipeline role and nothing downstream depends on its conventions.

- `top_k` networks come back ranked, best first, rather than as
  interchangeable samples.
- Residues forming the network are labelled (`reslabel`, default `hbnet`),
  which is what makes them selectable downstream.
- It returns glycine at every position outside the network.
  `restore_input_sequence` puts the input residues back, which is what makes
  the output usable as a design.
- Its 22 command line options are exposed through the one-dict pattern in
  section 2 rather than as individual attribute blocks.


## 11. Keeping a replacement backwards compatible

Existing XML protocols and driver scripts refer to movers by tag name, to
options by attribute name, to residues by label string, and to results by
score key. Each of those is an interface. Breaking one usually produces a
silently blank column rather than an error.

**Add attributes, do not change existing ones.** A new attribute must be
optional with a default that reproduces the previous behaviour. When
`reslabel_superimpose` was added to the `<RMSD>` tag it defaulted to the
measured residues, so every protocol already written kept working and kept
producing the same numbers.

**Do not rename or repurpose labels.** Downstream XML selects on the label
string. A label whose meaning shifts is worse than one that disappears,
because selectors keep matching and quietly measure the wrong residues.

**Score keys are part of the contract.** A driver script reads
`pose.cache` by exact key. Renaming a key blanks a CSV column with no
error: swapping ColabFold for Boltz2 in a pipeline that still read
`AF2_plddt` produced empty `plddt` and `scRMSD` columns while every other
column populated normally. When replacing a mover, update the key map in
the driver script in the same change, and prefer a single
column-to-key mapping over key strings scattered through the file.

**Same name, same meaning, same units - otherwise use a different name.**
Two movers in the same role reporting `plddt` on different scales is worse
than reporting `plddt` and `complex_plddt`.

**Keep the old mover registered.** Adding a replacement to
REGISTRATION_MODULES does not require removing what it replaces. Leaving
both means old protocols still parse, and the two can be compared on the
same inputs.

**State what cannot be preserved.** If a tool genuinely cannot honour part
of its role contract - renumbers residues, cannot fold a complex, has no
multi-output mode - say so in the mover header comment and name which
downstream metrics stop being valid as a result. That is more useful than a
mover that appears interchangeable and silently is not.

**Check a replacement against the previous mover output before trusting
it.** Run both on the same input, compare the score keys each produces and
the residue counts each labels. The offline replay in section 8 does this
without a GPU when the question is about labels or indices.
