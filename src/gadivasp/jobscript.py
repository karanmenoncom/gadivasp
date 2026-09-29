"""Drop-in for the local `jobscript.py` used in the notebooks:

    from gadivasp import jobscript
    jobscript.jobscript(ncpus=84, mem='60gb', walltime='4:00:00')

It (re)writes the PBS script that ASE_VASP_COMMAND points to, so the *next*
`atoms.get_potential_energy()` submits with those resources. That is the whole trick of the
pipeline: ASE writes the inputs, runs `qsub <script>` inside the calculation directory
(hence `#PBS -l wd` in the script), and the call returns as soon as the job is queued.
"""
import os
from pathlib import Path

from .config import load_config
from .jobs import render_jobscript


def jobscript(ncpus=None, mem=None, walltime=None, executable=None, cfg=None, path=None,
              set_env=True):
    """Write the shared job script and point ASE at it. Returns the script path."""
    cfg = cfg or load_config()
    path = Path(path or cfg.get("jobscript_path") or "~/jobscript.job").expanduser()
    render_jobscript(path.parent, "vasp", cfg, ncpus=ncpus, mem=mem, walltime=walltime,
                     executable=executable, filename=path.name)
    if set_env:
        os.environ["ASE_VASP_COMMAND"] = f"qsub {path}"
    return path
