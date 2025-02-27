#!/bin/bash

# Create the parent directory for emittance_scan
mkdir -p emittance_scan

# Loop through the CSV file line by line
# Skipping the header line using `tail -n +2` to avoid processing the first line
tail -n +2 emittance_and_beam_radius.csv | while IFS=',' read -r emittance beam_radius fraction; do
  
  # Create the subdirectory for each line with emittance, beam radius, and fraction
  dir="emittance_scan/emittance-${emittance}_radius-${fraction}"
  mkdir -p ${dir}  # Ensure directory exists
  
  # Create jobscript by replacing placeholder EMITTANCE with the current emittance
  sed "s/EMITTANCE/${emittance}/g; s/BEAM_RADIUS/${beam_radius}/g" base_jobscript.sh > ${dir}/jobscript.sh
  
  # Change to the newly created directory
  cd ${dir}
  
  # Submit the job using qsub
  qsub jobscript.sh 
  
  # Go back to the parent directory
  cd ../..

done

