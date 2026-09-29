"""Vacuum / cell-height convergence and extrapolation (notebook cells 93-172).

For charged slabs the energy converges slowly with cell height L, so the notebook fits
    E(L) = a / L^2 + b / L^3 + c        (or a / L^2 + c)
and takes c as the infinite-vacuum energy.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit


def model3(x, a, b, c):
    return a / x**2 + b / x**3 + c


def model2(x, a, c):
    return a / x**2 + c


@dataclass
class VacuumFit:
    params: tuple
    e_inf: float
    r2: float
    adj_r2: float
    chi2: float
    model: str

    def predict(self, x):
        f = model3 if self.model == "a/x^2 + b/x^3 + c" else model2
        return f(np.asarray(x, float), *self.params)


def fit_vacuum(heights, energies, terms=3):
    """Fit E(L). terms=3 -> a/L^2 + b/L^3 + c ; terms=2 -> a/L^2 + c. Returns VacuumFit."""
    x, y = np.asarray(heights, float), np.asarray(energies, float)
    f, label = (model3, "a/x^2 + b/x^3 + c") if terms == 3 else (model2, "a/x^2 + c")
    popt, _ = curve_fit(f, x, y, maxfev=10000)
    yfit = f(x, *popt)
    ss_res, ss_tot = np.sum((y - yfit) ** 2), np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, p = len(y), len(popt)
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else float("nan")
    chi2 = np.sum((y - yfit) ** 2 / yfit)          # as defined in the notebook
    return VacuumFit(tuple(popt), float(popt[-1]), float(r2), float(adj), float(chi2), label)
