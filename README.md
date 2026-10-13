# RosettaLink
RosettaLink: (Python) Movers to Link AI Models with Rosetta Workflows

## Dev installation
`pip install -e .`

## Caveats
Attribute values must not be empty strings. `work_dir=""` reports that `work_dir` is not a valid option for the mover, which is misleading. Omit the attribute instead; a mover given none makes its own temporary directory.

