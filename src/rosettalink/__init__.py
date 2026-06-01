from rosettalink.PyRosettaScripts import pyrosetta_scripts

def init(options='-ex1 -ex2aro', *, extra_options='', set_logging_handler=None, notebook=None, silent=False):
    """
    Initialize PyRosetta with the given options.

    Parameters:
    options (str): Options to initialize PyRosetta.
    extra_options (str): Additional options to append.
    set_logging_handler: Optional logging handler configuration.
    notebook: Optional notebook integration flag.
    silent (bool): Suppress PyRosetta output when True.
    """
    pyrosetta_scripts.init(
        options,
        extra_options=extra_options,
        set_logging_handler=set_logging_handler,
        notebook=notebook,
        silent=silent,
    )