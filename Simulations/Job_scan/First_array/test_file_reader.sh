#!/bin/bash --login
#$ -cwd
#$ -t 1-3
#$ -l short

# Read the comma-separated values
LINE_NUM=$((SGE_TASK_ID + 1))  # Skip header
EMITTANCE=$(awk -F',' "NR==$LINE_NUM {print \$1}" emittance_and_beam_radius.csv)
RADIUS=$(awk -F',' "NR==$LINE_NUM {print \$2}" emittance_and_beam_radius.csv)


# Print the extracted value for testing
echo "Emittance is $EMITTANCE and radius is $RADIUS"
