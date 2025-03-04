#!/bin/bash


for cores in 2 4 8 12 16 20 24 28 32; do
  dir="${cores}_cores"  # Store emX folders inside emittance_scan
  mkdir -p ${dir}  # Ensure directory exists
  sed "s/CORES/${cores}/g" base_jobscript.sh > ${dir}/jobscript.sh
  cd ${dir}
  # Link necessary files
  ln -s ~/scratch/qv3d_2.0/* ./
  ln -s ../ContinueBackFiles/* ./
  # Submit the job
  qsub jobscript.sh 
  
  cd ..  # Move back to the original directory
done
