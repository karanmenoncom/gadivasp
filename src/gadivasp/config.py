"""Configuration: everything user/cluster specific lives here, never in code."""
import os
from pathlib import Path
import yaml

DEFAULTS = {
    "project": None,
    "scratch_root": None,
    "queue": None,
    "modules": ["vasp/5.4.4"],
    "executable": "vasp_std",
    "storage": None,
    "default_resources": {"ncpus": 48, "mem": "50gb", "walltime": "1:00:00"},
    "chgsum_script": None,
    "bader_exe": None,
    "zval": {"Ni": 10, "N": 5, "H": 1, "O": 6, "Ce": 12},
}


def load_config(path=None):
    """Load config from `path`, $GADIVASP_CONFIG, ./config.yaml or ~/.gadivasp.yaml."""
    candidates = [path, os.environ.get("GADIVASP_CONFIG"), "config.yaml",
                  Path.home() / ".gadivasp.yaml"]
    cfg = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in DEFAULTS.items()}
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            with open(Path(c).expanduser()) as f:
                user = yaml.safe_load(f) or {}
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            cfg["_source"] = str(Path(c).expanduser())
            break
    return cfg
