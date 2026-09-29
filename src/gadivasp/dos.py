"""DOSCAR parsing and d-band descriptors (notebook cells 21 and 398)."""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def read_doscar(filename):
    """Return dict: energies (E - Ef), total, and per-atom {'s','p','d'} arrays.

    Handles spin-polarised (lorbit=11 -> 18 columns/atom; ispin=2 -> 2 total columns)
    by returning both spin channels as arrays of shape (nedos, nspin).
    """
    lines = open(filename).read().splitlines()
    natoms = int(lines[0].split()[0])
    hdr = lines[5].split()
    nedos, ef = int(hdr[2]), float(hdr[3])
    tot = np.array([[float(x) for x in lines[6 + i].split()] for i in range(nedos)])
    energies = tot[:, 0] - ef
    nspin_tot = (tot.shape[1] - 1) // 2 if tot.shape[1] in (3, 5) else 1
    atoms = {}
    block = 6 + nedos
    for a in range(natoms):
        start = block + a * (nedos + 1) + 1
        d = np.array([[float(x) for x in lines[start + i].split()] for i in range(nedos)])
        ncol = d.shape[1] - 1
        if ncol == 9:            # lorbit=11, spin-paired: s py pz px dxy dyz dz2 dxz dx2
            s, p, dd = d[:, 1], d[:, 2:5].sum(1), d[:, 5:10].sum(1)
            atoms[a] = {"s": s[:, None], "p": p[:, None], "d": dd[:, None]}
        elif ncol == 18:         # spin-polarised: up/down interleaved per orbital
            up, dn = d[:, 1::2], d[:, 2::2]
            atoms[a] = {"s": np.c_[up[:, 0], dn[:, 0]], "p": np.c_[up[:, 1:4].sum(1), dn[:, 1:4].sum(1)],
                        "d": np.c_[up[:, 4:9].sum(1), dn[:, 4:9].sum(1)]}
        else:
            atoms[a] = {"raw": d[:, 1:]}
    return {"energies": energies, "efermi": ef, "total": tot[:, 1:], "atoms": atoms}


def dband_stats(energies, dos, window=(-10.0, 5.0)):
    """d-band centre, width (sqrt of 2nd moment about 0, as in the notebook) and
    number of occupied states. `energies` are relative to E_F."""
    m = (energies > window[0]) & (energies < window[1])
    e, g = energies[m], dos[m]
    n_d = _trapz(g, e)
    centre = _trapz(e * g, e) / n_d
    width = float(np.sqrt(_trapz(e**2 * g, e) / n_d))
    occ = e <= 0.0
    return {"d_band_center": float(centre), "d_band_width": width,
            "n_states_total": float(n_d), "n_states_occupied": float(_trapz(g[occ], e[occ]))}
