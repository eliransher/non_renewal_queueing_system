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
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt


path = '/scratch/eliransc/non_renewal/training_batches/steady_1'
file_list = os.listdir(path)
data_paths = [os.path.join(path, name) for name in file_list]
len(data_paths)*128

corr_res_x = {}
corr_res_y = {}
for lower in np.linspace(-0.5, 0.4, 10):
    corr_res_x[lower] = np.array([])
    corr_res_y[lower] = np.array([])

for ind in tqdm(range(len(data_paths))):
    x,y = pkl.load(open(data_paths[ind], 'rb'))
    arr = x[:,10]
    for lower in np.linspace(-0.5,0.4,10):
        upper = lower + 0.1
        if lower > -0.15:
            if corr_res_x[lower].shape[0]> 0 :
                corr_res_x[lower] = np.concatenate((corr_res_x[lower], x[(arr<upper)&(arr>lower) ,:]), axis = 0)
                corr_res_y[lower] = np.concatenate((corr_res_y[lower], y[(arr<upper)&(arr>lower) ,:]), axis = 0)
            else:
                corr_res_x[lower] = x[(arr<upper)&(arr>lower) ,:]
                corr_res_y[lower] = y[(arr<upper)&(arr>lower) ,:]


for key in corr_res_y.keys():
    if key > -0.15:
        pkl.dump(corr_res_y[key][:, :1500], open('/scratch/eliransc/non_renewal/corr_res_steady_1_lower_'+str(key)+'_y.pkl', 'wb'))
        print(key)