#!/bin/bash

for em in 1 2 3; do
  dir=em${em}
  mkdir ${dir}
  sed "s/EMITTANCE/${em}/g" base.txt > ${dir}/new.txt
done

