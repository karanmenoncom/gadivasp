"""Reading energies and combining them into reaction / adsorption energies."""
from pathlib import Path
from ase.io import read


def read_energy(directory, name="vasprun.xml"):
    """Final potential energy (eV) of a finished calculation."""
    return read(Path(directory) / name, index=-1).get_potential_energy()


def read_atoms(directory, name="vasprun.xml"):
    return read(Path(directory) / name, index=-1)


def is_converged(directory):
    """True if OUTCAR reports reached required accuracy (relaxation) or finished ok."""
    p = Path(directory) / "OUTCAR"
    if not p.exists():
        return False
    txt = p.read_text(errors="ignore")
    return "reached required accuracy" in txt or "General timing and accounting" in txt


def reaction_energy(products, reactants):
    """Sum(coeff * E) over products minus reactants.

    Each side is a list of (coefficient, energy) pairs. The notebook wrote these by hand,
    e.g. '-433.606 - 2*425.827 + 3*421.917 + 19.53'.
    """
    return sum(c * e for c, e in products) - sum(c * e for c, e in reactants)


def adsorption_energy(e_total, e_slab, e_gas, n_gas=1):
    """E_ads = E(slab+ads) - E(slab) - n * E(gas).

    In the notebook the gas reference (E_NH3 ~ -19.53 eV) is the '+19.53' term.
    """
    return e_total - e_slab - n_gas * e_gas
