# Gadi setup

One-time setup for the ASE -> qsub workflow. Use the files in `cluster/` as templates.

1. **Environment.** Append `cluster/bashrc.example` to `~/.bashrc` (fill in `PROJECT`/`USER`) and
   `source ~/.bashrc`. The two variables that matter are `ASE_VASP_COMMAND` (the submit command) and
   `VASP_PP_PATH` (POTCAR root, laid out as `potpaw_PBE/<element>/POTCAR`).
2. **Config.** `cp config.example.yaml config.yaml` and set `project`, `scratch_root`, modules.
   `config.yaml` is git-ignored.
3. **Install.** `pip install --user -e .` (or in an ARE Jupyter session).
4. **Check.** `module load vasp/5.4.4 && which vasp_std`, and
   `ls $VASP_PP_PATH/potpaw_PBE/Ni/POTCAR`.

## Things worth knowing
* Calculations belong under `/scratch/PROJECT/USER`, not `$HOME` (small quota).
* Gadi compute nodes have no internet; run `git push` and `pip install` from a login node.
* `vasp/5.4.4` loads `openmpi/4.1.2` automatically; the job script pins the VASP version.
* Do not put a placeholder line such as `export LD_LIBRARY_PATH=/path/to/...` in `.bashrc`: each
  `LD_LIBRARY_PATH` export overwrites the previous one and points nowhere.
