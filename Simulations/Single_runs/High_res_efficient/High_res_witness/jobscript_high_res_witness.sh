#!/bin/bash --login
#$ -cwd                 # Job will run in the current directory (where you ran qsub)
#$ -pe amd.pe 32        # Choose a PE name from the tables below and a number of cores
#$ -l s_rt=03:00:00     # Sets job time limit
#$ -N qv3d              # Set the job's name

# Load any required modulefiles
module load libs/intel-18.0/hdf5/1.10.5_mpi
module load mpi/intel-18.0/openmpi/4.0.1


# Inform app how many cores to use
# For OpenMP applications (multicore but all in a single compute node):
#export OMP_NUM_THREADS=$NSLOTS
#mpirun qv3dMPIX.e v.ini

# For MPI applications (small jobs on a single node or larger jobs across multiple compute nodes)
mpirun -n $NSLOTS qv3dMPIX.e v_high_res_witness.ini
