import numpy as np
from ase.build import fcc111, molecule
import gadivasp as gv
from gadivasp import charged, structures, vacuum, jobs, presets
from gadivasp.dos import dband_stats
from gadivasp.bader import read_acf

CFG = {"project": "abc1", "queue": None, "modules": ["vasp/5.4.4"],
       "executable": "vasp_std", "storage": None, "zval": {"Ni": 10, "N": 5, "H": 1},
       "default_resources": {"ncpus": 48, "mem": "50gb", "walltime": "1:00:00"}}


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
    assert "module load vasp/5.4.4" in t and "storage" not in t and "#PBS -q" not in t
    t2 = jobs.render_jobscript(tmp_path, "x", dict(CFG, queue="normalbw", modules=["openmpi/4.1.2", "vasp/5.4.4"]), filename="j2.sh").read_text()
    assert "#PBS -q normalbw" in t2 and "module load openmpi/4.1.2 vasp/5.4.4" in t2


def test_qstat_parser():
    sample = ("\ngadi-pbs:\n" + " " * 60 + "Req'd Req'd Elap\n"
              "Job ID Username Queue Jobname SessID NDS TSK Memory Time S Time\n"
              "123456.gadi-pbs user1 normalbw slab_1e 4242 2 56 100gb 04:00 R 00:10\n"
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


def _fake_cluster(tmp_path, monkeypatch, qstat_line=""):
    """Fake `qsub` / `qstat` on PATH so the ASE-submits-the-job route can be tested."""
    _fake_potcars(tmp_path / "pp")
    monkeypatch.setenv("VASP_PP_PATH", str(tmp_path / "pp"))
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "qsub").write_text(f'#!/bin/bash\necho "$(pwd) $@" >> {tmp_path}/qsub.log\necho 424242.gadi-pbs\n')
    (bin_ / "qstat").write_text("#!/bin/bash\nprintf 'h1\\nh2\\nh3\\n'\n"
                                + (f"echo '{qstat_line}'\n" if qstat_line else ""))
    for f in bin_.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:" + __import__("os").environ["PATH"])
    return tmp_path / "qsub.log"


def test_get_potential_energy_route_submits_once(tmp_path, monkeypatch):
    log = _fake_cluster(tmp_path, monkeypatch)
    cfg = dict(CFG, scratch_root=str(tmp_path / "scratch"))
    slab = fcc111("Ni", size=(2, 2, 3), a=3.52, vacuum=6)          # note: pbc not set by hand
    c = gv.Calculation("ni_relax", slab, cfg, preset="relax", charge=1)
    jid = c.submit()                                                # atoms.get_potential_energy()
    assert jid == "424242.gadi-pbs"
    d = c.directory
    incar = (d / "INCAR").read_text()
    assert "NELECT = 121" in incar and "IBRION = 2" in incar
    assert {"POSCAR", "KPOINTS", "POTCAR", "job.sh", "manifest.json"} <= {p.name for p in d.iterdir()}
    assert str(d) in log.read_text() and "job.sh" in log.read_text()   # qsub ran INSIDE the folder
    assert "#PBS -l wd" in (d / "job.sh").read_text()


def test_no_double_submission(tmp_path, monkeypatch):
    log = _fake_cluster(tmp_path, monkeypatch,
                        qstat_line="424242.gadi-pbs user1 normalbw ni_relax 1 2 56 100gb 04:00 R 00:01")
    cfg = dict(CFG, scratch_root=str(tmp_path / "scratch"))
    c = gv.Calculation("ni_relax", fcc111("Ni", size=(2, 2, 3), a=3.52, vacuum=6), cfg)
    c.submit()
    assert c.status() == "Running"
    assert c.submit() == "active"                                   # already queued/running: not resubmitted
    assert log.read_text().count("\n") == 1
    c.submit(force=True)
    assert log.read_text().count("\n") == 2


def test_jobscript_dropin_sets_ase_command(tmp_path, monkeypatch):
    from gadivasp import jobscript
    p = jobscript.jobscript(ncpus=84, mem="60gb", walltime="4:00:00", cfg=CFG, path=tmp_path / "jobscript.job")
    assert "ncpus=84" in p.read_text()
    assert __import__("os").environ["ASE_VASP_COMMAND"] == f"qsub {p}"


def test_show_structure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from gadivasp.plotting import structure_views
    slab = fcc111("Ni", size=(2, 2, 3), a=3.52, vacuum=6)
    structure_views(slab + molecule("NH3")).savefig(tmp_path / "v.png")
    assert (tmp_path / "v.png").stat().st_size > 1000


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


def test_adsorb_atop():
    from gadivasp.structures import adsorb_atop
    slab = fcc111("Ni", size=(2, 2, 3), a=3.52, vacuum=6)
    ads = adsorb_atop(slab, molecule("NH3"), height=2.0)
    n = len(slab)
    assert ads.get_chemical_symbols()[n] == "N" and len(ads) == n + 4
    assert abs(ads.positions[n, 2] - slab.positions[:, 2].max() - 2.0) < 1e-6
