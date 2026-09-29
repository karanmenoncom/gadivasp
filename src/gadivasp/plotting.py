"""Plot helpers: consistent style, reaction profiles, vacuum fits, DOS."""
import numpy as np
import matplotlib.pyplot as plt


def set_style():
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 120, "savefig.bbox": "tight"})


def reaction_profile(steps, ax=None, label=None, color="black", width=1.0, gap=0.5, connect=True):
    """Step diagram like notebook cell 352. `steps` = list of energies (eV), one per state."""
    ax = ax or plt.gca()
    x = 0.0
    for i, e in enumerate(steps):
        ax.plot([x, x + width], [e, e], color=color, lw=2, label=label if i == 0 else None)
        if connect and i + 1 < len(steps):
            ax.plot([x + width, x + width + gap], [e, steps[i + 1]], color=color, ls="--", lw=0.6)
        x += width + gap
    ax.set_ylabel("Relative energy (eV)")
    ax.set_xticks([])
    return ax


def vacuum_fit_plot(heights, energies, fit, ax=None):
    ax = ax or plt.gca()
    xs = np.linspace(min(heights), max(heights), 200)
    ax.scatter(heights, energies, color="red", label="DFT")
    ax.plot(xs, fit.predict(xs), color="blue", label=f"fit: {fit.model}")
    ax.set_xlabel("Cell height along z (Å)")
    ax.set_ylabel("Total energy (eV)")
    ax.set_title(f"R² = {fit.r2:.5f}, E(∞) = {fit.e_inf:.4f} eV")
    ax.legend()
    return ax


def dos_plot(energies, dos_up, dos_down=None, ax=None, label=None):
    """Spin-resolved DOS, spin-down drawn mirrored (as in the notebook)."""
    ax = ax or plt.gca()
    ax.plot(energies, dos_up, label=label)
    if dos_down is not None:
        ax.plot(energies, -np.asarray(dos_down))
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("$E - E_F$ (eV)")
    ax.set_ylabel("Density of states")
    ax.set_yticks([0])
    return ax
