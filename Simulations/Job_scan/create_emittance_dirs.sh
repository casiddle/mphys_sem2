#!/bin/bash

for em in 1 2 3; do
  dir="em${em}"
  mkdir -p ${dir}  # Ensure directory exists
  sed "s/EMITTANCE/${em}/g" base_jobscript.sh > ${dir}/jobscript.sh
  cd ${dir}
  
  
  # Submit the job
  sbatch jobscript.sh  
  
  cd ..
done