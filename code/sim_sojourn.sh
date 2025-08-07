#!/bin/bash
#SBATCH --job-name=sample_ph            # Job name
#SBATCH --account=power-general-users
#SBATCH --time=22:00:00
#SBATCH --partition=power-general
#SBATCH --ntasks=1                      # Number of tasks per array job
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --cpus-per-task=1               # CPUs per task
#SBATCH --mem-per-cpu=4G                # Memory per CPU
#SBATCH --output=sample_ph%A_%a.out    # Output file: Job ID and array task ID
#SBATCH --error=sample_ph%A_%a.err     # Error file: Job ID and array task ID


source /scratch200/davidfine/queues/queues/bin/activate
python /a/home/cc/students/math/davidfine/non_renewal_queueing_system/code/sojourn_time_sim.py