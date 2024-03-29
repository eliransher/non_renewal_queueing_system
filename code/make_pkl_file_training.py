import numpy as np
import os
import sys
import pickle as pkl
import torch
from tqdm import tqdm

path = '/scratch/eliransc/non_renewal/depart_1'

files = os.listdir(path)

print('start')

for ind, file in tqdm(enumerate(files)):

    try:
        file_ = pkl.load(open(os.path.join(path, file), 'rb'))
        if ind == 0:
            inpt = file_[0].reshape(1, file_[0].shape[0])
            output = file_[1].reshape(1, file_[1].shape[0])
        else:
            inpt = np.concatenate((inpt, file_[0].reshape(1, file_[0].shape[0])), axis=0)
            output = np.concatenate((output, file_[1].reshape(1, file_[1].shape[0])), axis=0)
    except:
        print('bad input')

pkl.dump((inpt, output), open('/scratch/eliransc/non_renewal/pkl_training/depart_1_training.pkl', 'wb'))