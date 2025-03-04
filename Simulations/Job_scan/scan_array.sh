#!/bin/bash --login
#$ -cwd
#$ -t 1-3
#$ -pe amd.pe 4
#$ -l short

#Load required modulefiles
source /etc/profile

module purge
module load libs/intel-18.0/hdf5/1.10.5_mpi
module load mpi/intel-18.0/openmpi/4.0.1
module load apps/anaconda3/5.2.0/bin

export PATH=$HOME/.local/bin:$PATH

# Create emittance scan directories
mkdir -p emittance_scan/em-$SGE_TASK_ID
cd emittance_scan/em-$SGE_TASK_ID || exit 1

# Link necessary files
ln -sf ~/scratch/qv3d_2.0/* ./
ln -sf ../../ContinueBackFiles/* ./

# Create the witness beam
python pseedpar.py 60 120 7314 5.75 $SGE_TASK_ID 2e5 > wit.dat

# Run the simulation
mpirun -s $NSLOTS qv3dMPIX.e ../../v.ini
