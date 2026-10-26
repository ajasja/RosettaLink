# RosettaLink
RosettaLink: (Python) Movers to Link AI Models with Rosetta Workflows

## Dev installation
Temporarily, until RosettaLink is merged into PyRosetta, choose a [publicly available needed wheel](https://graylab.jhu.edu/download/PyRosetta4/archive/release/0.temp-rosettalink/) and follow these steps:

```sh
screen -D -RR PublicWheels
wget https://graylab.jhu.edu/download/PyRosetta4/archive/release/0.temp-rosettalink/pyrosetta-2025.29.post0.dev0+14.commits.6b923229bb-cp312-cp312-linux_x86_64.whl # Takes multiple hours
conda create --prefix ~/.conda/envs/RosettaLink
conda activate ~/.conda/envs/RosettaLink
conda install python=3.12
pip install pyrosetta-2025.29.post0.dev0+14.commits.6b923229bb-cp312-cp312-linux_x86_64.whl
git clone https://github.com/ajasja/RosettaLink.git
cd RosettaLink
pip install -e .
```

## Caveats
Attribute values must not be empty strings. `work_dir=""` reports that `work_dir` is not a valid option for the mover, which is misleading. Omit the attribute instead; a mover given none makes its own temporary directory.


## Wheel download links
These download links require graylab login credentials. If you need the file, contact us

Ubuntu wheel python 310
https://graylab.jhu.edu/download/PyRosetta4/archive/commits/PyRosetta4.Release.python310.ubuntu.wheel/pyrosetta-2025.29.post0.dev0+14.commits.6b923229bb-cp310-cp310-linux_x86_64.whl

Ubuntu wheel python 311 https://graylab.jhu.edu/download/PyRosetta4/archive/commits/PyRosetta4.Release.python311.ubuntu.wheel/pyrosetta-2025.29.post0.dev0+14.commits.6b923229bb-cp311-cp311-linux_x86_64.whl

Ubuntu wheel python 312
https://graylab.jhu.edu/download/PyRosetta4/archive/commits/PyRosetta4.Release.python312.ubuntu.wheel/pyrosetta-2025.29.post0.dev0+14.commits.6b923229bb-cp312-cp312-linux_x86_64.whl

Ubuntu wheel python 313
https://graylab.jhu.edu/download/PyRosetta4/archive/commits/PyRosetta4.Release.python313.ubuntu.wheel/pyrosetta-2025.29.post0.dev0+14.commits.6b923229bb-cp313-cp313-linux_x86_64.whl
