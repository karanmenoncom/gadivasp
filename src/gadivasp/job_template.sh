#!/bin/bash
#PBS -N {name}
#PBS -P {project}
#PBS -q {queue}
#PBS -l walltime={walltime}
#PBS -l ncpus={ncpus}
#PBS -l mem={mem}
#PBS -l software=vasp
#PBS -l wd
{storage_line}
module load {modules}

mpirun {executable} > vasp.log
