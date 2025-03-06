#!/bin/bash --login
#$ -cwd
#$ -t 1-3
#$ -pe amd.pe 32
#$ -l s_rt=00:30:00

#Load required modulefiles
source /etc/profile

module purge
module load libs/intel-18.0/hdf5/1.10.5_mpi
module load mpi/intel-18.0/openmpi/4.0.1
module load apps/anaconda3/5.2.0/bin

export PATH=$HOME/.local/bin:$PATH

# Extract emittance and radii from files
LINE_NUM=$((SGE_TASK_ID + 1))  # Skip header
EMITTANCE=$(awk -F',' "NR==$LINE_NUM {print \$1}" emittance_and_beam_radius.csv)
RADIUS=$(awk -F',' "NR==$LINE_NUM {print \$2}" emittance_and_beam_radius.csv)
FRACTION=$(awk -F',' "NR==$LINE_NUM {print \$3}" emittance_and_beam_radius.csv)


# Create emittance scan directories
dir="emittance_scan/emittance-${EMITTANCE}_radius-${FRACTION}"
mkdir -p ${dir}

# Change to new directory
cd ${dir} || exit 1

# Link necessary files
# Edit with your qv3dMPIX.e, pseedpar.py, and ContinueBack file locations
ln -sf ~/scratch/qv3d_2.0/* ./
ln -sf ~/scratch/ContinueBackFiles/* ./

# Create the witness beam
python pseedpar.py 60 120 7e14 $RADIUS $EMITTANCE 2e5 > wit.dat

# Run the simulation
mpirun -n $NSLOTS qv3dMPIX.e ../../v.ini
