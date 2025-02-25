#!/bin/bash

# Create the parent directory
mkdir -p emittance_radius_scan

# Skip the header and read emittance and radius values from CSV
tail -n +2 emittance_and_beam_radius.csv | while IFS=, read -r em radius; do
  dir="emittance_radius_scan/em${em}_r${radius}"  # Store directories inside emittance_scan
  mkdir -p "${dir}"  # Ensure directory exists


  # Replace placeholders in base_jobscript.sh and create a job script
  sed "s/EMITTANCE/${em}/g; s/RADIUS/${radius}/g" base_jobscript.sh > "${dir}/jobscript.sh"

  # Submit the job using qsub
  (cd "${dir}" && qsub jobscript.sh)

done
