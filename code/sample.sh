#!/bin/bash
#SBATCH -t 0-03:58
#SBATCH -A def-dkrass
#SBATCH --mem 20000
source /scratch200/davidfine/queues/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/non_renewal_queueing_system/code/batch_data_steady_0.py