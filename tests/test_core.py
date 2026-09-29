import numpy as np
from ase.build import fcc111, molecule
import gadivasp as gv
from gadivasp import charged, structures, vacuum, jobs, presets
from gadivasp.dos import dband_stats
from gadivasp.bader import read_acf

CFG = {"project": "abc1", "queue": "normalbw", "modules": ["openmpi/4.1.2", "vasp/5.4.4"],
       "executable": "vasp_std", "storage": None, "zval": {"Ni": 10, "N": 5, "H": 1},
       "default_resources": {"ncpus": 56, "mem": "100gb", "walltime": "4:00:00"}}


def test_nelect_matches_notebook():
    slab = fcc111("Ni", size=(4, 4, 5), a=3.52, vacuum=5, orthogonal=False)
    assert len(slab) == 80
    assert charged.nelect_for(slab, 1) == 801                      # notebook: slab 1e
    from ase import Atoms
    both = slab + molecule("NH3")
    assert charged.nelect_for(both, 1) == 809                      # notebook: NH3 on slab, 1e


def test_dipole_flags():
    p = presets.make_params("scf", dipole=True)
    assert p["ldipol"] is True and p["idipol"] == 3
    assert presets.make_params("relax", efield=0.5)["ldipol"] is True


def test_jobscript(tmp_path):
    f = jobs.render_jobscript(tmp_path, "slab_1e_relax", CFG, ncpus=28)
    t = f.read_text()
    assert "#PBS -P abc1" in t and "ncpus=28" in t and "-l wd" in t
    assert "module load openmpi/4.1.2 vasp/5.4.4" in t and "storage" not in t


def test_qstat_parser():
    sample = ("\ngadi-pbs:\n" + " " * 60 + "Req'd Req'd Elap\n"
              "Job ID Username Queue Jobname SessID NDS TSK Memory Time S Time\n"
              "123456.gadi-pbs km2447 normalbw slab_1e 4242 2 56 100gb 04:00 R 00:10\n"
              "   Job run at Tue\n")
    j = jobs.parse_qstat(sample)
    assert j[0]["status"] == "R" and j[0]["name"] == "slab_1e"


def test_vacuum_fit_recovers_c():
    x = np.array([18.0, 20.5, 23.0, 25.5, 28.0])
    y = -100 / x**2 - 500 / x**3 - 440.0
    fit = vacuum.fit_vacuum(x, y)
    assert abs(fit.e_inf + 440.0) < 1e-3 and fit.r2 > 0.999999


def test_dband():
    e = np.linspace(-10, 5, 3001)
    g = np.exp(-(e + 2) ** 2)
    s = dband_stats(e, g)
    assert abs(s["d_band_center"] + 2) < 0.01


def test_fix_layers():
    slab = fcc111("Ni", size=(2, 2, 5), a=3.52, vacuum=5)
    structures.fix_bottom_layers(slab, n_free_layers=3)
    assert len(slab.constraints[0].index) == 8                     # 2 fixed layers x 4 atoms


def _fake_potcars(root):
    for el, z in {"Ni": 10, "N": 5, "H": 1}.items():
        d = root / "potpaw_PBE" / el
        d.mkdir(parents=True)
        (d / "POTCAR").write_text(f"  PAW_PBE {el} 01Jan2000\n{z}.0000000000000000\n"
                                  f" parameters from PSCTR are:\n   TITEL  = PAW_PBE {el} 01Jan2000\n"
                                  f"   POMASS =   58.693; ZVAL   =   {z}.000    mass and valenz\n"
                                  "   ENMAX  =  270.000; ENMIN  =  202.500 eV\n End of Dataset\n")


def test_prepare_dry_run(tmp_path, monkeypatch):
    _fake_potcars(tmp_path / "pp")
    monkeypatch.setenv("VASP_PP_PATH", str(tmp_path / "pp"))
    cfg = dict(CFG, scratch_root=str(tmp_path))
    slab = fcc111("Ni", size=(2, 2, 3), a=3.52, vacuum=6)
    c = gv.Calculation("ni_relax", slab, cfg, preset="relax", charge=1)
    c.prepare(dry_run=True)
    incar = (c.directory / "INCAR").read_text()
    assert "NELECT = 121" in incar and "IBRION = 2" in incar
    assert (c.directory / "job.sh").exists() and (c.directory / "manifest.json").exists()


def test_acf(tmp_path):
    p = tmp_path / "ACF.dat"
    p.write_text("    #         X           Y           Z       CHARGE      MIN DIST   ATOMIC VOL\n"
                 " --------------------------------------------------------------------------------\n"
                 "    1    0.0   0.0   0.0    9.80   1.2   10.0\n")
    assert read_acf(p) == [9.80]


# ---- methods added from the Ni-NH3 / Data_Analysis / manuscript / CeO2 notebooks ----
def test_slab_from_bulk_sorted_and_fixable():
    from ase.build import bulk
    from gadivasp.structures import slab_from_bulk, fix_lateral
    slab = slab_from_bulk(bulk("Ni", "fcc", a=3.52, cubic=False), layers=5, repeat=(2, 2))
    z = slab.positions[:, 2]
    assert len(slab) == 20 and (z[1:] >= z[:-1] - 1e-9).all()      # bottom -> top ordering
    ads = slab + molecule("NH3")
    fix_lateral(ads, range(8), range(20, 24))
    assert len(ads.constraints) == 5


def test_ceo2_helpers():
    from gadivasp.ceo2 import ceo2_u_params, surface_energy
    p = ceo2_u_params()
    assert p["ldau_luj"]["Ce"] == {"L": 3, "U": 5.0, "J": 0.0} and p["lmaxmix"] == 6
    # notebook cell 136: n=6, 1x1 cell, 5.4073^2 A^2
    g = surface_energy(-151.86215143, -105.49638373 / 4, 6, 5.4073 ** 2)
    assert 0.5 < g < 3.0


def test_thermo():
    from gadivasp import thermo
    assert abs(thermo.corrected(-0.70, 0.10, 0.0) + 0.60) < 1e-12


def _fake_outcar(path, atoms, modes):
    """Write an OUTCAR fragment with `modes` = list of (freq_cm1, eigvec[natoms,3])."""
    lines = [f"   NIONS =  {len(atoms)}"]
    for k, (f, v) in enumerate(modes, 1):
        lines.append(f"  {k} f  =  100.0 THz   600.0 2PiTHz {f:.6f} cm-1   400.0 meV")
        lines.append("             X         Y         Z           dx          dy          dz")
        for p, d in zip(atoms.positions, v):
            lines.append("  " + " ".join(f"{x:.6f}" for x in (*p, *d)))
        lines.append("")
    path.write_text("\n".join(lines))


def test_vibrations_identify_symmetric_stretch(tmp_path):
    from gadivasp import vibrations as vb
    nh3 = molecule("NH3")
    n = nh3.get_positions()[0]
    sym = np.array([[0, 0, 0]] + [(p - n) / np.linalg.norm(p - n) for p in nh3.positions[1:]])
    asym = np.zeros((4, 3)); asym[1] = (nh3.positions[1] - n); asym[2] = -(nh3.positions[2] - n)
    _fake_outcar(tmp_path / "OUTCAR", nh3, [(3480.0, sym), (3400.0, asym), (200.0, sym)])
    f, v = vb.parse_modes(tmp_path / "OUTCAR")
    assert f.shape == (3,) and v.shape == (3, 4, 3)
    chars = vb.characterise_modes(f, v, nh3)
    found = vb.identify_modes(chars)
    assert found["N-H symmetric stretch (a1)"] == [0]
    assert 0 not in found.get("N-H asymmetric stretch (e)", [])


def test_bonded_metals_and_acf(tmp_path):
    from ase import Atoms
    from gadivasp.bader import read_bader, bonded_metals
    atoms = Atoms("Ni3N", positions=[[0, 0, 0], [2.5, 0, 0], [5, 0, 0], [1.2, 0, 1.6]], cell=[20, 20, 20], pbc=True)
    acf = tmp_path / "ACF.dat"
    acf.write_text("    #  X  Y  Z  CHARGE  MIN DIST  ATOMIC VOL\n" + "-" * 60 + "\n" +
                   "".join(f"{i+1} {p[0]} {p[1]} {p[2]} {q} 1.0 10.0\n" for i, (p, q) in
                           enumerate(zip(atoms.positions, [9.9, 9.9, 10.0, 5.6]))) + "-" * 60 + "\n VACUUM CHARGE: 0\n")
    df = read_bader(acf, atoms, {"Ni": 10, "N": 5})
    assert abs(df.loc[3, "CHARGE_TRANSFER"] + 0.6) < 1e-9
    assert list(bonded_metals(df).index) == [0, 1]                  # bridge site
