from pymcap import PyMCAP
import shutil


class McapCLI(PyMCAP):
    """Class to interact with the MCAP command-line interface (CLI) tool."""

    def _PyMCAP__get_executable(self):
        executable_path = shutil.which("mcap")
        if executable_path is None:
            return super().__get_executable()
        else:
            return executable_path
