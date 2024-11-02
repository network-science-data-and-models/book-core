#!/bin/bash
#SBATCH -J rumor_model                      # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                                # Number of tasks
#SBATCH --cpus-per-task 10                  # CPUs per task
#SBATCH -o output_%j.txt                    # Standard output file
#SBATCH -e error_%j.txt                     # Standard error file
#SBATCH --mail-user=smith.alyss@northeastern.edu  # Email
#SBATCH --mail-type=ALL                     # Type of email notifications

module load anaconda3/2022.05
module load openmpi/4.0.5                   # loading openMPI to run python MPI code
mpirun -np 10 /courses/PHYS7332.202510/shared/phys7332-env/bin/python mpi_rumor.py
