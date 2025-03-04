#!/bin/bash --login
#$ -cwd
#$ -t 1-5          # A job-array with 1000 "tasks", numbered 1...1000
                      # NOTE: No #$ -pe line so each task will use 1-core by default.

./myprog -in data.$SGE_TASK_ID.dat -out results.$SGE_TASK_ID.dat
               #
               # My input files are named: data.1.dat, data.2.dat, ..., data.1000.dat
               # 1000 tasks (copies of this job) will run.
               # Task 1 will read data.1.dat, task 2 will read data.2.dat, ... 
