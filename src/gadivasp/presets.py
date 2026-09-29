"""INCAR presets, taken from the parameters used in the Light-Ni notebook.

All presets share BASE (PBE, spin-polarised, Methfessel-Paxton N=1, sigma 0.2, ENCUT 500).
Override anything with keyword arguments: make_params("relax", kpts=[3, 3, 1]).
"""
import copy

BASE = dict(xc="pbe", kpts=[1, 1, 1], ismear=2, ispin=2, sigma=0.2, encut=500,
            ediff=1e-5, algo="Fast")

PRESETS = {
    # ionic relaxation, positions only (notebook cells 5, 24, 63)
    "relax": dict(nsw=200, ediffg=-0.05, ibrion=2, isif=2),
    # single point on a fixed geometry (notebook cell 70)
    "scf": dict(ibrion=-1),
    # finite-difference vibrations (IBRION=5), slab frozen so only adsorbate modes are computed.
    # potim 0.015 A in the charged studies, 0.02 A in the neutral Ni-NH3 study.
    "zpe": dict(ibrion=5, nsw=200, nfree=2, potim=0.015, lreal=False),
    # projected DOS on a denser k-grid (notebook cell 20). rwigs = Wigner-Seitz radius (A).
    "pdos": dict(ibrion=-1, kpts=[3, 3, 1], lorbit=11, lreal=False, rwigs=[1.23]),
    # AECCAR files for Bader analysis
    "bader": dict(ibrion=-1, laechg=True, lcharg=True, lreal=False),
    # Neutral-slab studies (Ni-NH3 notebook) used Gaussian smearing, sigma 0.1; the charged
    # 'light' studies used Methfessel-Paxton N=1, sigma 0.2. Both are kept as named variants.
    "relax_gauss": dict(nsw=200, ediffg=-0.05, ibrion=2, isif=2, ismear=0, sigma=0.1, lreal="Auto"),
    "zpe_gauss": dict(ibrion=5, nsw=200, nfree=2, potim=0.02, ismear=0, sigma=0.1, lreal="Auto"),
    # isolated molecule in a box (notebook cell 359)
    "molecule": dict(kpts=[1, 1, 1], ismear=0, ispin=1, sigma=0.1, nsw=200, ediffg=-0.01,
                     ibrion=2, isif=2, lreal="Auto"),
}


def make_params(preset, charge=0, neutral_nelect=None, dipole=False, efield=None, **overrides):
    """Return a dict of Vasp() keyword arguments.

    charge          extra electrons: NELECT = neutral_nelect + charge (notebook '1e','2e','3e')
    neutral_nelect  valence electron count of the neutral system (see charged.neutral_nelect)
    dipole          switch on the dipole correction along z (LDIPOL + IDIPOL=3)
    efield          external field in V/A along z; needs the dipole correction on
    """
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    p = copy.deepcopy(BASE)
    p.update(copy.deepcopy(PRESETS[preset]))
    if charge:
        if neutral_nelect is None:
            raise ValueError("charge != 0 needs neutral_nelect")
        p["nelect"] = neutral_nelect + charge
    if efield is not None:
        dipole = True
        p["efield"] = efield
    if dipole:
        # The notebook set idipol=3 without ldipol, and VASP ignores IDIPOL unless
        # LDIPOL = .TRUE. (same for EFIELD). Both are set here.
        p.update(ldipol=True, idipol=3)
    p.update(overrides)
    return p
