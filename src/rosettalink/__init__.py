from rosettalink.PyRosettaScripts import pyrosetta_scripts

def init(options='-ex1 -ex2aro', *, extra_options='', set_logging_handler=None, notebook=None, silent=False, config=None):
    """
    Initialize PyRosetta with the given options.

    Parameters:
    options (str): Options to initialize PyRosetta.
    extra_options (str): Additional options to append.
    set_logging_handler: Optional logging handler configuration.
    notebook: Optional notebook integration flag.
    silent (bool): Suppress PyRosetta output when True.
    config (str): Path to a rosettalink.config.yaml. When omitted, the file is
        looked for in the current directory and then in ~/.rosettalink/. Its
        contents are loaded into rosettalink.utils.configuration.
    """
    pyrosetta_scripts.init(
        options,
        extra_options=extra_options,
        set_logging_handler=set_logging_handler,
        notebook=notebook,
        silent=silent,
        config=config,
    )
