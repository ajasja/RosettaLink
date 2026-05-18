# RosettaLink
RosettaLink: (Python) Movers to Link AI Models with Rosetta Workflows

## Dev installation
`pip install -e .`

## Caveats
Values of provided attributes should _not_ be empty strings. For example, `work_dir=""` will throw an error that some attributes do not exist at all in the specification (`available options are: contig, delete_dir, extra_args, name, num_designs, rfdiffusion_path, ERROR: 'work_dir' is not a valid option for RFDiffusion`). Instead, omit the attribute to use its default value (it will automatically create `tempfile.TemporaryDirectory()`)

Flushing to stdout is inconsistent. Print statements will be printed out of order (for example, first you'll see debugs from RFDiffusion mover, then the result, and only then you'll see Rosetta initialisation info)

## Selecting residues from RfDiffusion
Select not _de novo_ designed residues with 
```xml
<StoredResidueSubset name="get_not_de_novo_residues" subset_name="inpaint_seq" />
```
or
```py
recall_selector = StoredResidueSubsetSelector('inpaint_seq')
recall_selector.apply(pose)
```

Selectors are named as in the .trb file. Currently exposed selectors:
* `inpaint_seq`: aminoacids whose identities were put into the input. What was taken in from the target (backbone + identity) = `True`. What was designed anew = `False`. Backbone from target, but the identities masked = `False` (when using the `contigmap.inpaint_seq=[A5-18]`) 
* `inpaint_str`: aminoacids whose backbone was taken as input. What was taken in from the target (backbone; identity doesn't matter) = `True`. What was designed anew = `False`. Backbone from target, but the identities masked = `True`.

You will probably want to negate the selector to be able to input the _de novo_ residues into the next step:
```xml
<Not name="get_de_novo_residues" selector="get_not_de_novo_residues" />
```