#!/bin/bash
#set a job name
#SBATCH --job-name=freedom
#a file for job output, you can check job progress
#SBATCH --output=logs/run1_%j.out
# a file for errors from the job
#SBATCH --error=logs/run1_%j.err
#time you think you need: default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#number of cores you are requesting
#SBATCH --cpus-per-task=20
#memory you are requesting
#SBATCH --mem=3Gb
#partition to use
#SBATCH --partition=short

module load anaconda3/2022.05
source activate FREEDOM

srun $1