#!/bin/bash

#SBATCH --nodes=1  #Allocate whatever you need here
#SBATCH --cpus-per-task=4  #Allocate whatever you need here 
#SBATCH --output=run.out
#SBATCH --job-name=test
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=aipi0122@colorado.edu
#SBATCH --mail-type=ALL

module purge
export DATA_DIR='/projects/aipi0122/dragg/dragg/data'
export LOGLEVEL='WARN'
source /curc/sw/anaconda3/2019.07/bin/activate
conda activate dragg
redis-server --daemonize yes
python -u main.py
