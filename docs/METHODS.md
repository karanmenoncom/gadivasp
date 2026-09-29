# Methods: what this pipeline does differently, and why

This is a record of the choices built into the workflow, each traceable to a notebook where it was
developed. It is the reason the repository exists: ASE and VASP do the calculations, but these
conventions are what make the results comparable across systems and charge states.

## 0. Submission through ASE (`jobscript`, `workflow.Calculation`)
Attach a `Vasp` calculator, call `atoms.get_potential_energy()`: ASE writes INCAR/KPOINTS/POSCAR/POTCAR
and runs the configured command (`qsub job.sh`) inside the calculation folder, so the whole
build -> set -> submit -> read -> plot cycle lives in one notebook. `jobscript.jobscript(ncpus, mem,
walltime)` sets the resources for the next submission. `#PBS -l wd` makes the job run where ASE
called `qsub`; the OUTCAR `FileNotFoundError` on return means "queued"; the job id is read back from
ASE's `vasp.out`. Repeat calls would resubmit, so the wrapper checks results and `qstat` first.

## 1. Slab construction with index conventions (`structures.slab_from_bulk`)
Slabs come from a *relaxed bulk* cell (Ni bulk at k = 28x28x28), cut with `surface()`, expanded
laterally, then **sorted by z**. Atom indices then run bottom to top, so "fix the bottom layers" is
`FixAtoms(range(n))`, and every adsorbate added later gets indices after the slab
(N = 80, H = 81..83 on the 80-atom Ni slab). All geometry descriptors and constraints use that.
*(Ni-NH3 notebook, cells 8-21.)*

## 2. Hot-electron model: charged slabs via NELECT (`charged`, `presets.make_params(charge=n)`)
Surface photo-excitation (LSPR hot electrons) is modelled by adding n = 1, 2, 3 electrons
(`NELECT = neutral + n`) to the slab and to every adsorbed state. `NELECT` is recomputed from the
composition each time (801 for the bare slab, 809 for slab + NH3), so it can never be stale.
*(Light-Ni notebook.)*

## 3. Charged-cell vacuum extrapolation (`vacuum.fit_vacuum`)
A charged periodic slab has an energy that depends on cell height L. Energies at several L are
fitted to `E(L) = a/L^2 + b/L^3 + c` (or `a/L^2 + c`); `c` is the infinite-vacuum energy. R^2,
adjusted R^2 and chi^2 are reported so a poor fit is visible. *(Light-Ni cells 93-172.)*

## 4. Dipole correction and electric field (`make_params(dipole=True, efield=...)`)
Sets `LDIPOL = .TRUE.`, `IDIPOL = 3`. `EFIELD` is only applied with the dipole correction on.

## 5. Adsorbate handling without lateral interaction (`structures.fix_lateral`, `drop_atom`)
For co-adsorbed states (NH2 + H, NH + H, N + H, 2N) the slab is frozen and each adsorbate atom is
constrained with `FixedLine` along z, so an atom can relax its height but cannot slide to another
site or into its periodic image. Seeds for NH2, NH, N come from clipping H atoms off relaxed NH3
(`drop_atom`). Several sites (atop, FCC, HCP) are tried and migration is checked.
*(Ni-NH3 cells 134-217, 262-273, 389-402.)*

## 6. Frozen-slab ZPE and gas-phase entropy (`presets['zpe']`, `zpe`, `thermo`)
Finite-difference frequencies (`IBRION = 5`, `NFREE = 2`, `POTIM = 0.015` or 0.02 A) with the slab
frozen, so the ZPE reflects the adsorbate only. `dZPE = ZPE(ads) - ZPE(gas)`; entropy enters
only through gas-phase species. Corrected energy: `dE + dZPE - T dS`.

## 7. NH3 mode analysis under charging (`vibrations`)
Modes are identified from the physics, not just frequency windows: N-H bond change and phase
(stretches), H-N-H angle change (scissoring), share of displacement on the NH3 atoms. Frequencies
are then compared with gas-phase values for the 0, 1e, 2e and 3e slabs, alongside H-N-H angle and
Ni-N distance changes. *(Data_Analysis notebook.)*

## 8. Bader analysis with automatic site detection (`bader`)
`chgsum.pl` builds `CHGCAR_sum` from AECCAR0 + AECCAR2 (the all-electron reference), Bader runs on
`CHGCAR` with `-ref`. `bonded_metals` finds the metal atoms bonded to N by geometry (all metals
within 1.25x the closest distance), so atop, bridge and hollow sites are handled without
hard-coded neighbour counts. Used for Ni, Cu and Ni/Cu sites. *(manuscript notebook.)*

## 9. DFT+U for CeO2 and surface energies (`ceo2`)
Dudarev U = 5 eV on Ce 4f (`LDAUTYPE = 2`, `LMAXMIX = 6`), restarting from the non-U wavefunctions.
`gamma = (E_slab - n E_bulk) / (2A)` in J/m^2. *(DFTU_CeO2 notebook.)*

## 10. Reaction profiles: light vs dark (`plotting.reaction_profile`)
Step diagrams with ZPE-corrected energies for the uncharged and charged pathways, overlaid.
