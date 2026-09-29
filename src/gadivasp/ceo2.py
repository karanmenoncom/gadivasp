"""CeO2 helpers: DFT+U settings and surface energies (DFTU_CeO2 notebook)."""

EV_PER_A2_TO_J_PER_M2 = 16.02176634


def ceo2_u_params(u_ce=5.0, j_ce=0.0, lmaxmix=6):
    """Dudarev DFT+U on Ce 4f (L=3). O gets no correction (L=-1). LMAXMIX=6 is needed for
    f-electron density mixing. Pass into Calculation(..., **ceo2_u_params()).
    Notebook: U=5.0 eV; ICHARG/ISTART=1 restarts from the non-U WAVECAR."""
    return dict(ldau=True, ldautype=2, ldauprint=1, lmaxmix=lmaxmix,
                ldau_luj={"Ce": {"L": 3, "U": u_ce, "J": j_ce}, "O": {"L": -1, "U": 0.0, "J": 0.0}})


def surface_energy(e_slab, e_bulk_per_fu, n_fu, area_A2, n_surfaces=2):
    """gamma = (E_slab - n * E_bulk_fu) / (n_surfaces * A), in J/m^2.

    The notebook computed this by hand, with a hardcoded area (5.4073^2 A^2 for a 1x1
    cell) and 16.02 as the eV/A^2 -> J/m^2 factor."""
    return (e_slab - n_fu * e_bulk_per_fu) / (n_surfaces * area_A2) * EV_PER_A2_TO_J_PER_M2
