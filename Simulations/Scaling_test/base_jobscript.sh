#!/bin/bash --login
#$ -cwd                 # Job will run in the current directory (where you ran qsub)
#$ -pe amd.pe CORES        # Number of cores determined in for loop
#$ -N qv3d              # Set the job's name

# Load any required modulefiles
source /etc/profile

module purge
module load libs/intel-18.0/hdf5/1.10.5_mpi
module load mpi/intel-18.0/openmpi/4.0.1
module load apps/anaconda3/5.2.0/bin

export PATH=$HOME/.local/bin:$PATH


# Inform app how many cores to use
# For OpenMP applications (multicore but all in a single compute node):
#export OMP_NUM_THREADS=$NSLOTS
#mpirun qv3dMPIX.e v.ini

# Create the witness beam
python pseedpar.py 60 120 7e14 5.75 2.0 2e5 > wit.dat

# For MPI applications (small jobs on a single node or larger jobs across multiple compute nodes)
mpirun -n $NSLOTS qv3dMPIX.e ../v.ini
