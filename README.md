# gadivasp

This repository documents and packages **my own DFT workflow** for studying ammonia decomposition on
charged (photo-excited) Ni, Cu and CeO2 surfaces with VASP on NCI Gadi. The point is not the ASE
wrappers; it is the set of conventions I developed so results are comparable across surfaces, sites
and charge states. They are written up in [`docs/METHODS.md`](docs/METHODS.md) and implemented in the package:

* **Charged-slab hot-electron model:** `NELECT` rebuilt from composition for each state, plus 1/L^2 + 1/L^3 vacuum extrapolation for charged cells.
* **Index-stable slabs:** built from relaxed bulk and sorted by z, so constraints and adsorbate indices are predictable.
* **Adsorbates without lateral interaction:** slab frozen, adsorbate atoms `FixedLine` along z, H clipped from NH3 to seed NH2/NH/N.
* **Frozen-slab ZPE and gas-phase entropy corrections** applied on top of DFT energies.
* **NH3 vibrational-mode identification** from eigenvectors (bond/angle response), used to track how charging shifts modes.
* **Bader with automatic site detection** (atop / bridge / hollow) using AECCAR-sum reference charge.
* **DFT+U CeO2** settings and surface energies.
* **Gadi plumbing:** PBS template, submission from the calculation folder, `qstat` monitoring, per-calculation manifest.

| Area | Module |
|---|---|
| Slabs, constraints, geometry | `structures` |
| INCAR presets, charge, dipole, field | `presets`, `charged` |
| Jobs on Gadi | `jobs`, `workflow` |
| Energies, vacuum fit, ZPE, thermo | `energies`, `vacuum`, `zpe`, `thermo` |
| Vibrational modes | `vibrations` |
| Bader, DOS / d-band | `bader`, `dos` |
| CeO2 + U | `ceo2` |
| Plots | `plotting` |

## Install (on Gadi or in ARE)

First-time cluster setup (bashrc, job script, POTCARs) is in [`docs/GADI_SETUP.md`](docs/GADI_SETUP.md) with templates in `cluster/`.


```bash
git clone https://github.com/<you>/gadivasp && cd gadivasp
pip install --user -e .
cp config.example.yaml config.yaml    # then edit project, scratch_root, modules
```

`config.yaml` is git-ignored. Nothing user- or project-specific lives in the code.
POTCARs are never included (VASP licence): point ASE at your own copy
(`VASP_PP_PATH`, or `[vasp] pp_path` in `~/.config/ase/config.ini`).

## How it works: ASE + VASP + Gadi in one call

The pipeline is ASE-native. `ASE_VASP_COMMAND` (or a per-calculation `command`) is `qsub job.sh`,
so asking a calculator-attached `Atoms` for its energy does everything:

```python
import gadivasp as gv
from gadivasp import jobscript, structures, plotting

cfg = gv.load_config()
jobscript.jobscript(ncpus=96, mem="100gb", walltime="4:00:00", cfg=cfg)   # PBS script + ASE_VASP_COMMAND

slab = structures.slab_from_bulk(ni_bulk, (1, 1, 1), layers=5, repeat=(4, 4))    # build
calc = gv.Calculation("slab_1e", slab, cfg, preset="relax", charge=+1)          # calculator setter
calc.submit()               # = atoms.get_potential_energy(): writes inputs, runs qsub in the folder
calc.status()               # Queued / Running / converged
plotting.structure_views(calc.atoms())                                           # visualise
```

`get_potential_energy()` returns as soon as the job is queued; the `FileNotFoundError` for OUTCAR
that ASE then raises simply means "not finished yet" and is handled. Plain ASE resubmits on every
call, so `submit()` checks the queue and the results first. `#PBS -l wd` in the job script is what
makes the job run in the folder ASE called `qsub` from.

See `examples/adsorption_cycle.ipynb` for slab creation, calculator, submission and visualisation end to end.

## Design notes

* One directory per calculation, with `INCAR/KPOINTS/POSCAR/POTCAR`, `job.sh` and `manifest.json`
  (job id, parameters, timestamps) so every result can be traced back.
* Jobs are submitted through ASE itself (`get_potential_energy()` runs `qsub` in the calculation
  directory), with a guard against duplicate submission.
* The VASP module version is pinned in `config.yaml` (`vasp/5.4.4`).

## Tests

```bash
pip install -e ".[dev]" && pytest
```

## Licence

MIT. VASP itself and its pseudopotentials are separately licensed and not distributed here.
