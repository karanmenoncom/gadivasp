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

```bash
git clone https://github.com/<you>/gadivasp && cd gadivasp
pip install --user -e .
cp config.example.yaml config.yaml    # then edit project, scratch_root, modules
```

`config.yaml` is git-ignored. Nothing user- or project-specific lives in the code.
POTCARs are never included (VASP licence): point ASE at your own copy
(`VASP_PP_PATH`, or `[vasp] pp_path` in `~/.config/ase/config.ini`).

## Minimal example

```python
import gadivasp as gv
from gadivasp.structures import ni111_slab, fix_bottom_layers

cfg  = gv.load_config()
slab = fix_bottom_layers(ni111_slab(size=(4, 4, 5), total_height=18.0), n_free_layers=3)

relax = gv.Calculation("slab_1e_relax", slab, cfg, preset="relax", charge=+1,
                       resources=dict(ncpus=56, mem="100gb", walltime="4:00:00"))
relax.prepare()
relax.submit()
gv.wait_for([relax.job_id])
print(relax.energy(), relax.converged())
```

See `examples/` for a notebook covering a full adsorption-energy cycle and a vacuum-convergence fit.
`sample_data/` holds tiny synthetic inputs so the analysis code runs without cluster access.

## Design notes

* One directory per calculation, with `INCAR/KPOINTS/POSCAR/POTCAR`, `job.sh` and `manifest.json`
  (job id, parameters, timestamps) so every result can be traced back.
* Jobs are submitted with `qsub` from inside the calculation directory (`#PBS -l wd`), not through
  ASE's blocking `get_potential_energy()`, so `ASE_VASP_COMMAND` is not needed.
* Modules are pinned (`openmpi/4.1.2`, `vasp/5.4.4`) in `config.yaml`.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

## Licence

MIT. VASP itself and its pseudopotentials are separately licensed and not distributed here.
