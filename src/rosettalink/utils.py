import os
from pyrosetta.rosetta.basic import Tracer
from pyrosetta.rosetta.basic import TracerPriority

def run_and_log(command, tracer_info, tracer_error):
    """Runs a command using os.system and also logs the command before running using print"""
    tracer_info << f"Running command: {command} \n" and tracer_info.flush()
    stat = os.system(command)
    wife = os.WIFEXITED(stat)
    exitCode = os.waitstatus_to_exitcode(stat)
    tracer_info << f"Command exited with status {stat} and WIFEXITED {wife}. Exit code: {exitCode} \n" and tracer_info.flush()
    if exitCode != 0:
        tracer_error << f" There was an error running the command. We consider it fatal to prevent any file loss. Check the logs and contact the developer. \n" and tracer_error.flush()
        dodatek = ""

        raise Exception(f" Command exited with exit code {exitCode}\n\n{dodatek}")

def setup_tracer(mover_name):
    new_tracer_fatal = Tracer(mover_name, TracerPriority.t_fatal)
    new_tracer_error = Tracer(mover_name, TracerPriority.t_error)
    new_tracer_warning = Tracer(mover_name, TracerPriority.t_warning)
    new_tracer_info = Tracer(mover_name, TracerPriority.t_info)
    new_tracer_debug = Tracer(mover_name, TracerPriority.t_debug)
    new_tracer_trace = Tracer(mover_name, TracerPriority.t_trace)
    return new_tracer_fatal, new_tracer_error, new_tracer_warning, new_tracer_info, new_tracer_debug, new_tracer_trace

