"""Slab, molecule and adsorbate helpers (from the structure cells of the notebook)."""
import numpy as np
from ase.build import fcc111, molecule, surface
from ase.build.supercells import make_supercell
from ase.build.tools import sort
from ase.constraints import FixAtoms, FixedLine


def ni111_slab(size=(4, 4, 5), a=3.524, vacuum=None, total_height=None):
    """Ni(111) slab. The notebook slab is 4x4x5 (80 Ni atoms)."""
    slab = fcc111("Ni", size=size, a=a, vacuum=(vacuum if vacuum is not None else 0.0),
                  orthogonal=False)
    if total_height is not None:
        set_cell_height(slab, total_height)
    return slab


def set_cell_height(atoms, height):
    """Set the c-axis length to `height` (A) and centre the atoms (notebook: set_cell + center).

    Vacuum sweeps in the notebook add 2.5/5/7.5/10 A to a base height of 18.0255 A.
    """
    cell = atoms.get_cell().copy()
    cell[2] = [0.0, 0.0, height]
    atoms.set_cell(cell, scale_atoms=False)
    atoms.center(axis=2)
    return atoms


def fix_bottom_layers(atoms, n_free_layers=3, tol=0.5):
    """Fix everything except the top `n_free_layers` z-layers of the slab.

    Adsorbate atoms above the slab would count as layers, so call this on the clean slab
    or pass `slab_indices` via fix_indices instead. Notebook: FixAtoms(0..79) on 5 layers
    with the top 3 relaxed ('top-3-relax') fixed only the bottom layers.
    """
    z = atoms.positions[:, 2]
    zs = np.sort(z)
    layer_tops = [zs[0]]
    for v in zs[1:]:
        if v - layer_tops[-1] > tol:
            layer_tops.append(v)
        else:
            layer_tops[-1] = v
    cutoff = layer_tops[-n_free_layers] - tol
    atoms.set_constraint(FixAtoms(mask=z < cutoff))
    return atoms


def fix_indices(atoms, indices):
    atoms.set_constraint(FixAtoms(indices=list(indices)))
    return atoms


def gas_molecule(name, box=10.0):
    """Molecule centred in a cubic periodic box (notebook cells 358, 365)."""
    m = molecule(name)
    m.set_cell((box, box, box))
    m.set_pbc(True)
    m.center()
    return m


def adsorption_geometry(atoms, metal_layer_indices, adsorbate_N, n_h_pairs=()):
    """Return the descriptors printed in the notebook: height above the metal layer,
    M-N distance to the closest metal atom, and N-H distances.

    metal_layer_indices : indices of the topmost metal layer (notebook: atoms 64..79)
    adsorbate_N         : index of the N atom (notebook: 80)
    n_h_pairs           : iterable of (N_index, H_index) pairs
    """
    z_layer = atoms.positions[list(metal_layer_indices), 2].mean()
    d = atoms.get_distances(adsorbate_N, list(metal_layer_indices), mic=True)
    out = {"height_A": float(atoms.positions[adsorbate_N, 2] - z_layer),
           "d_M_N_A": float(d.min()), "closest_metal": int(list(metal_layer_indices)[int(d.argmin())])}
    for n, h in n_h_pairs:
        out[f"d_{n}-{h}_A"] = float(atoms.get_distance(n, h, mic=True))
    return out


def slab_from_bulk(bulk, miller=(1, 1, 1), layers=5, vacuum=5.0, repeat=(2, 2)):
    """Slab from a *relaxed bulk* cell, then a lateral supercell, then atoms sorted by z.

    Sorting by z is the key convention of this pipeline: atom indices then run bottom to
    top, so "fix the bottom layers" is just FixAtoms(range(n_fixed)) and every adsorbate
    appended later gets indices after the slab (the notebooks' hard-coded 80, 81, 82...).
    """
    slab = surface(bulk, miller, layers, vacuum=vacuum)
    slab.set_pbc(True)
    slab = make_supercell(slab, [[repeat[0], 0, 0], [0, repeat[1], 0], [0, 0, 1]])
    return sort(slab, tags=slab.positions[:, 2])


def fix_lateral(atoms, fixed_indices, adsorbate_indices):
    """Freeze the slab and let the adsorbate move only along z (FixedLine).

    Used to get isolated-adsorbate energies without lateral interaction between periodic
    images (e.g. NH2 and H placed on the same cell): the adsorbate can relax its height but
    cannot slide to a different site or towards its neighbour.
    """
    cons = [FixAtoms(indices=list(fixed_indices))]
    cons += [FixedLine(int(i), direction=[0, 0, 1]) for i in adsorbate_indices]
    atoms.set_constraint(cons)
    return atoms


def drop_atom(atoms, index=-1):
    """Remove one atom, e.g. clip an H off adsorbed NH3 to seed an NH2 calculation."""
    new = atoms.copy()
    new.set_constraint(atoms.constraints)
    del new[index]
    return new


def check_forces(atoms, threshold=0.03):
    """True if any force component exceeds `threshold` eV/A (manuscript notebook)."""
    f = atoms.get_forces()
    return bool((abs(f) > threshold).any()), float(abs(f).max())


def adsorb_atop(slab, adsorbate, on_atom=None, height=2.0, anchor=0):
    """Place `adsorbate` (Atoms) above a slab atom (default: the highest one), atop site.

    `anchor` is the index of the adsorbate atom that sits at `height` above the site (N for
    NH3). Adsorbate indices follow the slab, e.g. N = len(slab), H = len(slab)+1... Keeps the
    slab's constraints."""
    from ase.build import add_adsorbate
    z = slab.positions[:, 2]
    i = int(z.argmax()) if on_atom is None else on_atom
    new = slab.copy()
    new.set_constraint(slab.constraints)
    add_adsorbate(new, adsorbate, height=height, position=tuple(slab.positions[i, :2]),
                  mol_index=anchor)
    return new
