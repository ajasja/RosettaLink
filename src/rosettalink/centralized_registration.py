# @file pyrosetta_scripts/centralized_registration.py
# @brief Central register of external components for PyRosettaScripts.
# @author Moritz Ertelt

import importlib

# base directory
BASE_DIR = 'rosettalink'

# Dictionary to organize modules by project directories with relative paths
REGISTRATION_MODULES = {
    'movers': [
        'RFDiffusion',
    ],
}


def register_all():
    for directory, modules in REGISTRATION_MODULES.items():
        for relative_module in modules:
            module_name = f"{BASE_DIR}.{directory}.{relative_module}"
            try:
                importlib.import_module(module_name)
                print(f"Successfully imported {module_name}")
            except ImportError as e:
                print(f"Optional module {module_name} could not be imported: {e}")
