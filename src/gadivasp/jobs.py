"""PBS job script rendering, submission and monitoring for Gadi."""
import re
import subprocess
import time
from importlib import resources
from pathlib import Path

_STATUS = {"Q": "Queued", "R": "Running", "E": "Exiting", "F": "Finished",
           "H": "Held", "S": "Suspended", "X": "Subjob finished"}


def render_jobscript(directory, name, cfg, ncpus=None, mem=None, walltime=None,
                     executable=None, filename="job.sh"):
    """Write a job script into `directory` and return its path.

    Each calculation keeps its own copy, so the exact resources and modules used are
    recorded next to the results. #PBS -l wd makes the job run where qsub was called.
    """
    res = cfg["default_resources"]
    if not cfg.get("project"):
        raise ValueError("config: `project` (NCI project code) is not set")
    template = resources.files("gadivasp").joinpath("job_template.sh").read_text()
    storage = cfg.get("storage")
    text = template.format(
        name=re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:15],   # PBS truncates long names
        project=cfg["project"], queue=cfg["queue"],
        walltime=walltime or res["walltime"], ncpus=ncpus or res["ncpus"],
        mem=mem or res["mem"], executable=executable or cfg["executable"],
        modules=" ".join(cfg["modules"]),
        storage_line=f"#PBS -l storage={storage}" if storage else "")
    text = "\n".join(l for l in text.splitlines() if l.strip() != "") + "\n"
    path = Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def submit(directory, script="job.sh"):
    """qsub `script` from inside `directory` (required because of `-l wd`). Returns job id."""
    out = subprocess.run(["qsub", script], cwd=directory, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"qsub failed in {directory}: {out.stderr.strip()}")
    return out.stdout.strip()          # e.g. '12345678.gadi-pbs'


_JOB = re.compile(r"^(\d+)\.gadi-pbs\b")


def parse_qstat(text):
    """Parse `qstat -swx` output (ported from the notebook). Returns list of dicts."""
    lines = text.strip().split("\n")
    jobs, i = [], 3                       # first three lines are the header
    while i < len(lines):
        if _JOB.match(lines[i]):
            parts = lines[i].split()
            if len(parts) >= 10:
                details = ""
                if i + 1 < len(lines) and not _JOB.match(lines[i + 1]):
                    details = lines[i + 1].strip()
                    i += 1
                jobs.append({"job_id": parts[0], "name": parts[3], "status": parts[9],
                             "description": _STATUS.get(parts[9], parts[9]), "details": details})
        i += 1
    return jobs


def job_status(job_id=None):
    """Status of all your jobs (or one). Finished jobs drop out of the list."""
    out = subprocess.run(["qstat", "-swx"], capture_output=True, text=True).stdout
    jobs = parse_qstat(out)
    if job_id:
        jobs = [j for j in jobs if job_id.split(".")[0] in j["job_id"]]
    return jobs


def wait_for(job_ids, interval=60, timeout=None, verbose=True):
    """Block until none of `job_ids` is queued or running any more."""
    ids = {j.split(".")[0] for j in job_ids}
    t0 = time.time()
    while True:
        active = [j for j in job_status() if j["job_id"].split(".")[0] in ids
                  and j["status"] in ("Q", "R", "E", "H", "S")]
        if verbose:
            print(time.strftime("%H:%M:%S"), {j["job_id"]: j["description"] for j in active})
        if not active:
            return
        if timeout and time.time() - t0 > timeout:
            raise TimeoutError(f"still active after {timeout}s: {[j['job_id'] for j in active]}")
        time.sleep(interval)
