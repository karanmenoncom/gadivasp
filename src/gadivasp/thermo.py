"""Corrections applied on top of DFT energies (ZPE and gas-phase entropy)."""
from .zpe import zpe


def corrected(delta_e, delta_zpe=0.0, t_delta_s=0.0):
    """dG ~ dE + dZPE - T dS   (all eV).

    Notebook practice: the slab is frozen for the frequency run, so ZPE terms come from
    adsorbate modes only; only gas-phase species carry an entropy term (T*S at the
    reaction temperature), adsorbates are treated as having negligible entropy.
    """
    return delta_e + delta_zpe - t_delta_s


def delta_zpe(adsorbed_dir, gas_dir=None, n_gas=1):
    """ZPE(adsorbed) - n * ZPE(gas). Returns (dZPE, n_imaginary_adsorbed)."""
    z_ads, n_imag = zpe(adsorbed_dir)
    z_gas = zpe(gas_dir)[0] * n_gas if gas_dir else 0.0
    return z_ads - z_gas, n_imag
