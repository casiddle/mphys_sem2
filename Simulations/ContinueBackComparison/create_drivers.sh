#!/bin/bash --login
#$ -cwd
#$ -t 1-99
#$ -pe amd.pe 20
#$ -l s_rt=00:05:00

#Load required modulefiles
source /etc/profile

module purge
module load libs/intel-18.0/hdf5/1.10.5_mpi
module load mpi/intel-18.0/openmpi/4.0.1
module load apps/anaconda3/5.2.0/bin

export PATH=$HOME/.local/bin:$PATH

# Extract emittance and radii from files
LINE_NUM=$((SGE_TASK_ID + 1))  # Skip header 
CB_POINT=$(awk -F',' "NR==$LINE_NUM {print \$1}" CB_point_array.csv)
UNROUNDED_POINT=$(awk -F',' "NR==$LINE_NUM {print \$3}" CB_point_array.csv)

# Create driver directories
dir="drivers/driver-${UNROUNDED_POINT}"
mkdir -p ${dir}

# Create input deck
sed "s/CB_POINT/${CB_POINT}/g" base_driver_v.ini > ${dir}/v.ini

# Change to new directory
cd ${dir} || exit 1

# Link necessary files
# Edit with your qv3dMPIX.e, pseedpar.py, and ContinueBack file locations
ln -sf ~/scratch/qv3d_2.0/* ./

# Run the simulation
mpirun -n $NSLOTS qv3dMPIX.e v.ini
