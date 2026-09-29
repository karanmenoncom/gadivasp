"""Calculation: one directory = one calculation (inputs, job, results, metadata)."""
import json
import time
from pathlib import Path

from ase.calculators.vasp import Vasp
from ase.io import write

from .jobs import render_jobscript, submit as _submit, job_status
from .energies import read_energy, read_atoms, is_converged
from .presets import make_params


class Calculation:
    """
    calc = Calculation("slab_1e_relax", atoms, cfg, preset="relax", charge=1)
    calc.prepare()          # writes INCAR/KPOINTS/POSCAR/POTCAR + job.sh + manifest.json
    calc.submit()           # qsub from inside the folder, returns the job id
    calc.status(); calc.energy(); calc.atoms()
    """

    def __init__(self, name, atoms, cfg, preset="relax", charge=0, neutral_nelect=None,
                 dipole=False, efield=None, resources=None, executable=None, parent=None, **params):
        self.name, self.atoms_in, self.cfg = name, atoms.copy(), cfg
        if atoms.constraints:
            self.atoms_in.set_constraint(atoms.constraints)
        root = Path(cfg["scratch_root"]) if cfg.get("scratch_root") else Path.cwd()
        self.directory = (root / (parent or "") / name).expanduser()
        self.resources = resources or {}
        self.executable = executable
        if charge and neutral_nelect is None:
            from .charged import neutral_nelect as nn
            neutral_nelect = nn(self.atoms_in, cfg.get("zval"))
        self.params = make_params(preset, charge=charge, neutral_nelect=neutral_nelect,
                                  dipole=dipole, efield=efield, **params)
        self.preset, self.charge = preset, charge
        self.job_id = self._manifest().get("job_id")

    # -- files ---------------------------------------------------------------
    def _manifest_path(self):
        return self.directory / "manifest.json"

    def _manifest(self):
        p = self._manifest_path()
        return json.loads(p.read_text()) if p.exists() else {}

    def prepare(self, dry_run=False):
        """Write all inputs. dry_run=True skips POTCAR (no VASP_PP_PATH needed), for testing."""
        self.directory.mkdir(parents=True, exist_ok=True)
        calc = Vasp(directory=str(self.directory), **self.params)
        if dry_run:
            calc.initialize(self.atoms_in)
            calc.write_incar(self.atoms_in, directory=str(self.directory))
            calc.write_kpoints(atoms=self.atoms_in, directory=str(self.directory))
            write(self.directory / "POSCAR", self.atoms_in, format="vasp", direct=True)
        else:
            calc.write_input(self.atoms_in)
        render_jobscript(self.directory, self.name, self.cfg, executable=self.executable,
                         **self.resources)
        self._save({"name": self.name, "preset": self.preset, "charge": self.charge,
                    "params": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                               for k, v in self.params.items()},
                    "prepared": time.strftime("%Y-%m-%d %H:%M:%S")})
        return self.directory

    def _save(self, extra):
        m = self._manifest()
        m.update(extra)
        self._manifest_path().write_text(json.dumps(m, indent=2))

    # -- running -------------------------------------------------------------
    def submit(self):
        if not (self.directory / "job.sh").exists():
            self.prepare()
        self.job_id = _submit(self.directory)
        self._save({"job_id": self.job_id, "submitted": time.strftime("%Y-%m-%d %H:%M:%S")})
        return self.job_id

    def status(self):
        if not self.job_id:
            return "not submitted"
        jobs = job_status(self.job_id)
        if jobs:
            return jobs[0]["description"]
        return "done" if is_converged(self.directory) else "finished (check OUTCAR)"

    # -- results -------------------------------------------------------------
    def energy(self):
        return read_energy(self.directory)

    def atoms(self):
        return read_atoms(self.directory)

    def converged(self):
        return is_converged(self.directory)
