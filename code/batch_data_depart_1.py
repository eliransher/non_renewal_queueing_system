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

file_name_used = []
cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]
path = '/scratch/eliransc/non_renewal/depart_1'
files = os.listdir(path)
true_files = [file for file in files if 'multi' in file]
batch_size = 128

path_dump_data_depart_1 = '/scratch/eliransc/non_renewal/training_batches/depart_1'
num_batches = int(len(true_files)/batch_size)


for batch_num in tqdm(range(len(os.listdir(path_dump_data_depart_1)), num_batches)):

    input_depart_1 = np.array([])
    output_depart_1 = np.array([])
    for batch_ind in range(batch_size):
        file_num = batch_num * batch_size + batch_ind
        try:
            inp, out = pkl.load(open(os.path.join(path, true_files[file_num]), 'rb'))
        except:
            print('keep the same')
        file_name_used.append(true_files[file_num])
        if batch_ind > 0:
            input_depart_1 = np.concatenate((input_depart_1, inp.reshape(1, inp.shape[0])), axis=0)
            output_depart_1 = np.concatenate((output_depart_1, out.reshape(1, out.shape[0]), ), axis=0)
        else:
            input_depart_1 = inp.reshape(1, inp.shape[0])
            output_depart_1 = out.reshape(1, out.shape[0])

    batch_name = 'depart_1_from_'+cluster_name+'_batch_num_' + str(batch_num)+'.pkl'

    pkl.dump((input_depart_1, output_depart_1), open(os.path.join(path_dump_data_depart_1, batch_name), 'wb'))
    pkl.dump(file_name_used, open('/scratch/eliransc/non_renewal/file_used_depart_1.pkl', 'wb'))