#!/bin/bash

# Create the parent directory
mkdir -p emittance_scan

for em in 1 2 3; do
  dir="emittance_scan/em${em}"  # Store emX folders inside emittance_scan
  mkdir -p ${dir}  # Ensure directory exists
  sed "s/EMITTANCE/${em}/g" base_jobscript.sh > ${dir}/jobscript.sh
  cd ${dir}
  
  # Submit the job
  qsub jobscript.sh 
  
  cd ../..  # Move back to the original directory
done
