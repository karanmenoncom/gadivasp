"""Calculation: one directory = one calculation, submitted the ASE way.

    calc = Calculation("slab_1e_relax", atoms, cfg, preset="relax", charge=1)
    calc.submit()      # atoms.get_potential_energy() -> ASE writes inputs + runs `qsub`
    calc.status()      # queued / running / done
    calc.energy()      # reads vasprun.xml once finished

`submit()` is the notebook idiom `atoms.get_potential_energy()`: the call writes INCAR, KPOINTS,
POSCAR, POTCAR, runs the qsub command inside the directory and returns as soon as the job is
queued (ASE then fails to find OUTCAR; that FileNotFoundError is the expected 'queued' signal
and is handled here). Plain ASE would submit again on every call, so `submit()` first checks
whether the calculation is already finished or still in the queue.
"""
import json
import time
from pathlib import Path

from ase.calculators.calculator import CalculatorError
from ase.calculators.vasp import Vasp

from .jobs import render_jobscript, job_status, job_id_from_output
from .energies import read_energy, read_atoms, is_converged
from .presets import make_params

_ACTIVE = ("Q", "R", "E", "H", "S")


class Calculation:
    def __init__(self, name, atoms, cfg, preset="relax", charge=0, neutral_nelect=None,
                 dipole=False, efield=None, resources=None, executable=None, parent=None, **params):
        self.name, self.cfg = name, cfg
        self.atoms_in = atoms.copy()
        if atoms.constraints:
            self.atoms_in.set_constraint(atoms.constraints)
        self.atoms_in.set_pbc(True)            # Vasp() refuses non-periodic cells
        root = Path(cfg["scratch_root"]) if cfg.get("scratch_root") else Path.cwd()
        self.directory = (root / (parent or "") / name).expanduser()
        self.resources, self.executable = resources or {}, executable
        if charge and neutral_nelect is None:
            from .charged import neutral_nelect as nn
            neutral_nelect = nn(self.atoms_in, cfg.get("zval"))
        self.params = make_params(preset, charge=charge, neutral_nelect=neutral_nelect,
                                  dipole=dipole, efield=efield, **params)
        self.preset, self.charge = preset, charge
        self.job_id = self._manifest().get("job_id")

    # -- bookkeeping ---------------------------------------------------------
    def _manifest(self):
        p = self.directory / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _save(self, **extra):
        self.directory.mkdir(parents=True, exist_ok=True)
        m = self._manifest()
        m.update(extra)
        (self.directory / "manifest.json").write_text(json.dumps(m, indent=2, default=str))

    # -- the ASE part --------------------------------------------------------
    def calculator(self):
        """The ASE Vasp calculator, attached to a copy of the atoms (calculator setter).
        The per-calculation job script is written next to the inputs, so the exact resources
        and modules stay with the results."""
        self.directory.mkdir(parents=True, exist_ok=True)
        render_jobscript(self.directory, self.name, self.cfg, executable=self.executable,
                         **self.resources)
        atoms = self.atoms_in.copy()
        atoms.set_constraint(self.atoms_in.constraints)
        calc = Vasp(directory=str(self.directory), command="qsub job.sh", atoms=atoms, **self.params)
        return calc, atoms

    def write_inputs(self):
        """Write INCAR/KPOINTS/POSCAR/POTCAR/job.sh without submitting (to inspect first)."""
        calc, atoms = self.calculator()
        calc.write_input(atoms)
        return self.directory

    def submit(self, force=False):
        """Submit via `atoms.get_potential_energy()`. Returns 'done', 'active' or the job id."""
        if not force:
            if self.converged():
                return "done"
            if self.is_active():
                return "active"
        calc, atoms = self.calculator()
        try:
            atoms.get_potential_energy()       # writes inputs, runs `qsub job.sh` in the folder
        except (FileNotFoundError, CalculatorError):
            pass                               # expected: the job is queued, OUTCAR does not exist yet
        self.job_id = job_id_from_output(self.directory)
        if not self.job_id:
            raise RuntimeError(f"no job id in {self.directory}/vasp.out; did qsub run?")
        self._save(name=self.name, preset=self.preset, charge=self.charge, job_id=self.job_id,
                   params=self.params, submitted=time.strftime("%Y-%m-%d %H:%M:%S"))
        return self.job_id

    # -- state ---------------------------------------------------------------
    def is_active(self):
        return bool(self.job_id) and any(j["status"] in _ACTIVE for j in job_status(self.job_id))

    def status(self):
        if not self.job_id:
            return "converged" if self.converged() else "not submitted"
        jobs = job_status(self.job_id)
        if jobs:
            return jobs[0]["description"]
        return "converged" if self.converged() else "finished (check OUTCAR)"

    def converged(self):
        return is_converged(self.directory)

    def energy(self):
        return read_energy(self.directory)

    def atoms(self):
        return read_atoms(self.directory)
