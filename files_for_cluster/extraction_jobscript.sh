#!/bin/bash 
#$ -cwd                 # Job will run in the current directory (where you ran qsub)
#$ -N parameter-extraction           # Sets the job's name
#$ -l short     # Sets job time limit

source /etc/profile

# Load any required modulefiles
module purge
module load apps/anaconda3/5.2.0/bin

# Activate the specific Conda environment
source activate my_env 

# Add .local/bin to PATH to ensure packages are found
export PATH=$HOME/.local/bin:$PATH

# Debugging: Print Python and SciPy version
python3 -c "import sys; import scipy; print('Python:', sys.version); print('SciPy:', scipy.__version__)"
python3 ./parameter_extraction.py

conda deactivate