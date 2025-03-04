#!/bin/bash

# Create the parent directory
mkdir -p emittance_scan

for em in 1 2 3; do
  dir="emittance_scan/em${em}"  # Store emX folders inside emittance_scan
  mkdir -p ${dir}  # Ensure directory exists
  sed "s/EMITTANCE/${em}/g" base_jobscript.sh > ${dir}/jobscript.sh
  cd ${dir}
  # Link necessary files
  ln -s ~/scratch/qv3d_2.0/* ./
  ln -s ../../ContinueBackFiles/* ./
  # Submit the job
  qsub jobscript.sh 
  
  cd ../..  # Move back to the original directory
done
