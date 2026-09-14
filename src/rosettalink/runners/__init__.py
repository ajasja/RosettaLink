# @file runners/__init__.py
# @brief Standalone scripts invoked as subprocesses by the movers.
#
# Nothing here imports rosettalink or pyrosetta: these scripts run in the
# python environment that has the wrapped tool installed, which is usually
# not the environment running PyRosetta.
