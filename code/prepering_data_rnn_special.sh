#!/bin/bash
#SBATCH -t 0-22:58
#SBATCH -A def-dkrass
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=40000M
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/RNN_queue/rnn-lstm-gru/prepering_data_RNN_special.py

