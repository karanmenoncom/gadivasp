"""Charged-slab bookkeeping (the '1e / 2e / 3e' calculations).

NELECT is an absolute number in VASP, so it has to be rebuilt every time the
composition changes (slab -> slab+NH3 -> slab+NH2+H ...). The notebook did this by hand
(801, 809, 808 ...); here it is derived from the atoms.
"""
from .config import DEFAULTS


def neutral_nelect(atoms, zval=None):
    zval = zval or DEFAULTS["zval"]
    missing = sorted({s for s in atoms.get_chemical_symbols() if s not in zval})
    if missing:
        raise KeyError(f"no valence electron count for {missing}; add them to `zval` in config")
    return sum(zval[s] for s in atoms.get_chemical_symbols())


def nelect_for(atoms, extra_electrons, zval=None):
    """NELECT for `atoms` carrying `extra_electrons` additional electrons."""
    return neutral_nelect(atoms, zval) + extra_electrons
