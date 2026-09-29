"""Zero-point energy from a VASP finite-difference frequency run (IBRION=5/6)."""
import re
from pathlib import Path

_F = re.compile(r"^\s*\d+\s+f(/i)?\s*=\s*([\d.]+)\s+THz\s+[\d.]+\s+2PiTHz\s+[\d.]+\s+cm-1\s+([\d.]+)\s+meV")


def read_frequencies(directory):
    """Return list of (meV, is_imaginary) from OUTCAR."""
    out = []
    for line in (Path(directory) / "OUTCAR").read_text(errors="ignore").splitlines():
        m = _F.match(line)
        if m:
            out.append((float(m.group(3)), m.group(1) is not None))
    return out


def zpe(directory, drop_imaginary=True):
    """ZPE in eV = 1/2 * sum(h nu) over real modes. Also returns number of imaginary modes."""
    freqs = read_frequencies(directory)
    if not freqs:
        raise ValueError(f"no frequencies in {directory}/OUTCAR (was IBRION=5/6 used?)")
    real = [e for e, imag in freqs if not imag]
    n_imag = sum(1 for _, imag in freqs if imag)
    return 0.5 * sum(real) / 1000.0, n_imag
