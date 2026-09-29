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


ELEMENT_COLORS = {"Ni": "lightyellow", "Cu": "orange", "N": "skyblue", "H": "lightgreen",
                  "O": "red", "C": "lightgrey", "Ce": "gold", "Pt": "silver", "Au": "gold"}


def show_structure(atoms, rotation="-90x,0y,0z", colors=None, title=None, ax=None, axes_arrows=True):
    """One view of a structure with your colour scheme and small x/y/z arrows (side view by
    default; use rotation='0x,0y,0z' for the top view)."""
    from ase.visualize.plot import plot_atoms
    ax = ax or plt.figure(figsize=(5, 5)).gca()
    cols = colors or [ELEMENT_COLORS.get(s, "grey") for s in atoms.get_chemical_symbols()]
    plot_atoms(atoms, ax, rotation=rotation, colors=cols, show_unit_cell=2)
    if axes_arrows:
        x0, y0 = ax.get_xlim()[0] + 1, ax.get_ylim()[0] + 1
        horiz, vert = ("x", "z") if rotation.startswith("-90x") else ("x", "y")
        ax.arrow(x0, y0, 1, 0, head_width=0.4, head_length=0.6, fc="red", ec="red")
        ax.arrow(x0, y0, 0, 1, head_width=0.4, head_length=0.6, fc="blue", ec="blue")
        ax.text(x0 + 1.2, y0 - 0.3, horiz, fontweight="bold")
        ax.text(x0 - 0.5, y0 + 1.3, vert, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def structure_views(atoms, title=None):
    """Side and top view next to each other."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    show_structure(atoms, "-90x,0y,0z", ax=axs[0], title=title)
    show_structure(atoms, "0x,0y,0z", ax=axs[1])
    return fig
