"""Bader charge analysis: chgsum.pl (AECCAR0+AECCAR2) -> bader -ref -> ACF.dat."""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def run_chgsum(directory, chgsum_script, aeccar0="AECCAR0", aeccar2="AECCAR2"):
    r = subprocess.run([str(Path(chgsum_script).expanduser()), aeccar0, aeccar2],
                       cwd=directory, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


def run_bader(directory, bader_exe, chgcar="CHGCAR", ref="CHGCAR_sum"):
    """Note: the notebook passed ' -ref ' (with spaces) as one argument, which bader
    would not parse as an option. Arguments are passed separately here."""
    r = subprocess.run([str(Path(bader_exe).expanduser()), chgcar, "-ref", ref],
                       cwd=directory, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


def read_acf(path):
    """Bader-assigned electron count per atom from ACF.dat (notebook: read_charges)."""
    charges = []
    for line in Path(path).read_text().splitlines():
        if re.match(r"\s*#", line) or re.match(r"\s*-", line):
            continue
        parts = line.split()
        if len(parts) == 7:
            charges.append(float(parts[4]))
    return charges


def net_charges(acf_charges, atoms, zval):
    """Net charge per atom = ZVAL - Bader electrons (positive = electron-poor)."""
    return [zval[s] - q for s, q in zip(atoms.get_chemical_symbols(), acf_charges)]


def read_bader(acf_file, atoms, valence):
    """ACF.dat as a DataFrame aligned with `atoms` (manuscript notebook: read_bader).

    Columns: CHARGE (Bader electrons), CHARGE_TRANSFER = ZVAL - CHARGE (positive = electron
    loss), DIST_TO_N (minimum-image distance to the adsorbate N, if there is exactly one)."""
    rows = []
    for line in Path(acf_file).read_text().splitlines():
        p = line.split()
        if len(p) == 7 and p[0].isdigit():
            rows.append([float(x) for x in p[1:]])
    df = pd.DataFrame(rows, columns=["X", "Y", "Z", "CHARGE", "MIN_DIST", "ATOMIC_VOL"])
    if len(df) != len(atoms):
        raise ValueError(f"{acf_file} has {len(df)} atoms, structure has {len(atoms)}: different runs?")
    df["ELEMENT"] = atoms.get_chemical_symbols()
    df["VALENCE"] = df["ELEMENT"].map(valence)
    df["CHARGE_TRANSFER"] = df["VALENCE"] - df["CHARGE"]
    n = [i for i, s in enumerate(df["ELEMENT"]) if s == "N"]
    df["DIST_TO_N"] = atoms.get_distances(n[0], range(len(atoms)), mic=True) if n else np.nan
    return df


def bonded_metals(df, metals=("Ni", "Cu"), tol=1.25, max_neighbors=None):
    """Metal atoms actually bonded to N: all metals within tol x (closest distance).

    Finds the adsorption site from the geometry (atop = 1, bridge = 2, hollow = 3) instead of
    assuming a fixed count of neighbours."""
    m = df[df["ELEMENT"].isin(metals)]
    if m.empty:
        return m
    nb = m[m["DIST_TO_N"] <= m["DIST_TO_N"].min() * tol].sort_values("DIST_TO_N")
    return nb.head(max_neighbors) if max_neighbors else nb
