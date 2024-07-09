from fastai.vision.all import *
from fastbook import *
from sklearn.model_selection import train_test_split
matplotlib.rc('image', cmap='Greys')
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import torch
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from scipy.special import factorial
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]
path = '/scratch/eliransc/non_renewal/full_system'

if not os.path.exists(path):
    os.mkdir(path)
files = os.listdir(path)

true_files = [file for file in files if 'multi' in file]
batch_size = 128

path_dump_data_depart_0 = '/scratch/eliransc/non_renewal/training_batches/full_system'
if not os.path.exists(path_dump_data_depart_0):
    os.mkdir(path_dump_data_depart_0)

num_batches = int(len(true_files) / batch_size)

for batch_num in tqdm(range(len(os.listdir(path_dump_data_depart_0)), num_batches)):

    input_full = np.array([])
    output_depart_full = np.array([])
    output_steady_full = np.array([])

    for batch_ind in range(batch_size):
        file_num = batch_num * batch_size + batch_ind
        inp, out_depart, out_steady = pkl.load(open(os.path.join(path, true_files[file_num]), 'rb'))
        out_depart = np.concatenate((out_depart[0], out_depart[1]))
        out_steady = np.concatenate((out_steady[0], out_steady[1]))

        if batch_ind > 0:
            input_full = np.concatenate((input_full, inp.reshape(1, inp.shape[0])), axis=0)
            output_depart_full = np.concatenate((output_depart_full, out_depart.reshape(1, out_depart.shape[0])),
                                                axis=0)
            output_steady_full = np.concatenate((output_steady_full, out_steady.reshape(1, out_steady.shape[0])),
                                                axis=0)
        else:
            input_full = inp.reshape(1, inp.shape[0])
            output_depart_full = out_depart.reshape(1, out_depart.shape[0])
            output_steady_full = out_steady.reshape(1, out_steady.shape[0])

    batch_name = 'full_system_from_' + cluster_name + '_batch_num_' + str(batch_num) + '.pkl'

    pkl.dump((input_full, output_depart_full, output_steady_full),
             open(os.path.join(path_dump_data_depart_0, batch_name), 'wb'))