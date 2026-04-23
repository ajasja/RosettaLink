# RosettaLink
RosettaLink: (Python) Movers to Link AI Models with Rosetta Workflows

## Dev installation
`pip install -e .`

## Caveats
Values of provided attributes should _not_ be empty strings. For example, `work_dir=""` will throw an error that some attributes do not exist at all in the specification (`available options are: contig, delete_dir, extra_args, name, num_designs, rfdiffusion_path, ERROR: 'work_dir' is not a valid option for RFDiffusion`). Instead, omit the attribute to use its default value (it will automatically create `tempfile.TemporaryDirectory()`)

Flushing to stdout is inconsistent. Print statements will be printed out of order (for example, first you'll see debugs from RFDiffusion mover, then the result, and only then you'll see Rosetta initialisation info)

