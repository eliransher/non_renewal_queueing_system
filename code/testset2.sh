#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 20000
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/proje cts/def-dkrass/eliransc/non_renewal_queueing_system/code/sim_test2_take2.py