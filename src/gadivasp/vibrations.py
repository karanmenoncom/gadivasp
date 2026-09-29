"""Vibrational-mode analysis of adsorbed NH3 from a VASP finite-difference (IBRION=5) OUTCAR.

Ported from the Data_Analysis notebook. VASP gives frequencies and displacement
eigenvectors for the (free) adsorbate only, because the slab is frozen. To see how surface
charging changes the *adsorbed* molecule, each mode is classified as an NH3 internal mode
from how much it changes N-H bond lengths and H-N-H angles, rather than by frequency alone:

    symmetric stretch (a1) : large, in-phase bond changes, no angle change
    asymmetric stretch (e) : large, out-of-phase bond changes (degenerate pair)
    scissoring        (e)  : angle change, little bond change (degenerate pair)
    wagging / umbrella (a1): mixed angle + bond change, mostly NH3 motion

The frequency windows and thresholds are the ones used in the notebook (`CRITERIA`).
"""
import itertools
import re
import numpy as np

GAS_PHASE_CM1 = {"N-H symmetric stretch (a1)": 3534, "N-H asymmetric stretch (e)": 3464,
                 "H-N-H scissoring (e)": 1765, "N-H wagging (a1)": 1139}

CRITERIA = {   # freq window (cm-1) and thresholds, as in the notebook
    "N-H symmetric stretch (a1)":  dict(window=(3300, 3700), min_bond=0.005, min_sym=0.7, max_angle=2.0),
    "N-H asymmetric stretch (e)":  dict(window=(3300, 3700), min_bond=0.005, max_sym=0.7, max_angle=2.0),
    "H-N-H scissoring (e)":        dict(window=(1500, 2000), min_angle=1.0, max_bond=0.01),
    "N-H wagging (a1)":            dict(window=(1000, 1300), min_angle=0.5, min_bond=0.001, min_nh3=20),
}

_FREQ = re.compile(r"^\s*\d+\s+f(/i)?\s*=.*?([\d.]+)\s+cm-1")


def parse_modes(outcar):
    """Return (frequencies_cm1, eigenvectors[nmodes, natoms, 3]). Imaginary modes are negative."""
    lines = open(outcar, errors="ignore").read().splitlines()
    natoms = next(int(l.split("NIONS =")[1].split()[0]) for l in lines if "NIONS =" in l)
    freqs, vecs = [], []
    i = 0
    while i < len(lines):
        m = _FREQ.match(lines[i])
        if m:
            f = float(m.group(2))
            freqs.append(-f if m.group(1) else f)
            j = i + 1
            while "dx" not in lines[j]:
                j += 1
            vecs.append(np.array([[float(x) for x in lines[j + 1 + a].split()[3:6]] for a in range(natoms)]))
            i = j + natoms
        i += 1
    return np.array(freqs), np.array(vecs)


def find_nh3_groups(atoms):
    """Each N with its three nearest H atoms (minimum-image distances)."""
    sym = np.array(atoms.get_chemical_symbols())
    h = np.where(sym == "H")[0]
    groups = []
    for n in np.where(sym == "N")[0]:
        d = atoms.get_distances(n, h, mic=True)
        groups.append({"N": int(n), "H": [int(x) for x in h[np.argsort(d)[:3]]]})
    return groups


def nh3_contribution(vec, groups):
    """% of the summed atomic displacement that sits on the N and H atoms of the NH3 group(s)."""
    norms = np.linalg.norm(vec, axis=1)
    idx = [g["N"] for g in groups] + [h for g in groups for h in g["H"]]
    return 100 * norms[idx].sum() / norms.sum() if norms.sum() > 0 else 0.0


def _bond_angle_response(vec, groups, atoms, step=0.02):
    """Mean |d(N-H)| (A), symmetry of bond changes (1 = all in phase), variance, mean d(HNH) (deg)."""
    pos = atoms.get_positions()
    disp = pos + step * vec / max(np.abs(vec).max(), 1e-12)
    bond, sym, var, ang = [], [], [], []
    for g in groups:
        n, hs = g["N"], g["H"]
        d0 = np.array([np.linalg.norm(pos[h] - pos[n]) for h in hs])
        d1 = np.array([np.linalg.norm(disp[h] - disp[n]) for h in hs])
        db = d1 - d0
        bond.append(np.abs(db).mean())
        sym.append(abs(db.sum()) / (np.abs(db).sum() + 1e-12))
        var.append(db.var())
        a0, a1 = [], []
        for i, j in itertools.combinations(range(3), 2):
            for p, store in ((pos, a0), (disp, a1)):
                u, v = p[hs[i]] - p[n], p[hs[j]] - p[n]
                store.append(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        ang.append(np.abs(np.array(a1) - np.array(a0)).mean())
    return dict(avg_bond_change=np.mean(bond), bond_symmetry=np.mean(sym),
                bond_variance=np.mean(var), avg_angle_change=np.mean(ang))


def characterise_modes(freqs, vecs, atoms, min_freq=500, min_nh3=10):
    """Per-mode descriptors for modes above `min_freq` with >`min_nh3`% NH3 character."""
    groups = find_nh3_groups(atoms)
    out = {}
    for i, (f, v) in enumerate(zip(freqs, vecs)):
        if f < min_freq:
            continue
        c = nh3_contribution(v, groups)
        if c > min_nh3:
            out[i] = dict(frequency=float(f), nh3_contribution=c, **_bond_angle_response(v, groups, atoms))
    return out


def identify_modes(chars, criteria=CRITERIA):
    """Assign characterised modes to NH3 mode names. Returns {name: [mode indices]}."""
    found = {}
    for name, c in criteria.items():
        lo, hi = c["window"]
        cand = []
        for i, d in chars.items():
            ok = lo < d["frequency"] < hi
            ok &= d["avg_bond_change"] > c.get("min_bond", -1) and d["avg_bond_change"] < c.get("max_bond", 1e9)
            ok &= d["avg_angle_change"] > c.get("min_angle", -1) and d["avg_angle_change"] < c.get("max_angle", 1e9)
            ok &= d["bond_symmetry"] > c.get("min_sym", -1) and d["bond_symmetry"] < c.get("max_sym", 1e9)
            ok &= d["nh3_contribution"] > c.get("min_nh3", -1)
            if ok:
                cand.append(i)
        key = {"symmetric": "avg_bond_change", "asymmetric": "bond_variance",
               "scissoring": "avg_angle_change", "wagging": "nh3_contribution"}
        k = next(v for s, v in key.items() if s in name.lower())
        cand.sort(key=lambda i: chars[i][k], reverse=True)
        n = 2 if "(e)" in name else 1
        if cand:
            found[name] = cand[:n]
    return found


def mode_shifts(found, chars, reference=GAS_PHASE_CM1):
    """Mean frequency per identified mode and its shift from the gas-phase value (cm-1)."""
    return {name: (float(np.mean([chars[i]["frequency"] for i in idx])),
                   float(np.mean([chars[i]["frequency"] for i in idx])) - reference[name])
            for name, idx in found.items()}
