import os

def run_and_log(mover_name, command):
    """Runs a command using os.system and also logs the command before running using print"""
    stat = os.system(command)
    wife = os.WIFEXITED(stat)
    exitCode = os.waitstatus_to_exitcode(stat)
    print(f"[{mover_name}] Command exited with status {stat} and WIFEXITED {wife}. Exit code: {exitCode}")
    if exitCode != 0:
        print(f"[{mover_name}] There was an error running the command. We consider it fatal to prevent any file loss. Check the logs and contact the developer.")
        dodatek = ""

        raise Exception(f"[{mover_name}] Command exited with exit code {exitCode}\n\n{dodatek}")
