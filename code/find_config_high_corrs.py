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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sys

# for cedar
# corr_res_x, _ = pkl.load( open('/home/eliransc/projects/def-dkrass/eliransc/Notebooks/non_renewal/non_renewal/corr_res_depart_0.pkl', 'rb'))

# for narval
corr_res_x, _ = pkl.load( open('/scratch/eliransc/corr_res_depart_0.pkl', 'rb'))


df_moms = pd.DataFrame([])
key = 0.4
for key in corr_res_x.keys():
    if key > -20.25:
        arrival_sec_mom = np.exp(corr_res_x[key][:, 1])
        arrival_third_mom = np.exp(corr_res_x[key][:, 2])
        arrival_fourth_mom = np.exp(corr_res_x[key][:, 3])
        ser_first_mom = np.exp(corr_res_x[key][:, 10])
        ser_second_mom = np.exp(corr_res_x[key][:, 11])
        ser_third_mom = np.exp(corr_res_x[key][:, 12])
        ser_fourth_mom = np.exp(corr_res_x[key][:, 13])

        for ind in range(min(1000, corr_res_x[key].shape[0])):
            curr_ind = df_moms.shape[0]
            df_moms.loc[curr_ind, 'key'] = key
            df_moms.loc[curr_ind, 'arrival_secmom'] = arrival_sec_mom[ind]
            df_moms.loc[curr_ind, 'arrival_third_mom'] = arrival_third_mom[ind]
            df_moms.loc[curr_ind, 'arrival_fourth_mom'] = arrival_fourth_mom[ind]
            df_moms.loc[curr_ind, 'ser_first_mom'] = ser_first_mom[ind]
            df_moms.loc[curr_ind, 'ser_second_mom'] = ser_second_mom[ind]
            df_moms.loc[curr_ind, 'ser_third_mom'] = ser_third_mom[ind]
            df_moms.loc[curr_ind, 'ser_fourth_mom'] = ser_fourth_mom[ind]


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Data creation

df = df_moms
# Encode the group labels
label_encoder = LabelEncoder()
df['group_encoded'] = label_encoder.fit_transform(df['key'])

# Split the data
X = df[['arrival_secmom', 'arrival_third_mom', 'arrival_fourth_mom','ser_first_mom', 'ser_second_mom', 'ser_third_mom', 'ser_fourth_mom']]
y = df['group_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

def get_ph():
    if sys.platform == 'linux':
        path = '/scratch/eliransc/ph_examples'
    else:
        path = r'C:\Users\user\workspace\data\ph_random\ph_mean_1'

    files = os.listdir(path)

    ind_file = np.random.randint(len(files))

    data_all = pkl.load(open(os.path.join(path, files[ind_file]), 'rb'))

    ind_file1 = np.random.randint(len(data_all))

    data = data_all[ind_file1]
    return data

from numpy.linalg import matrix_power
from scipy.special import factorial

def ser_moment_n(s, A, mom):
    '''
    ser_moment_n
    :param s:
    :param A:
    :param mom:
    :return:
    '''
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) * factorial(mom) * np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False


def compute_first_n_moments(s, A, n=3):
    '''
    compute_first_n_moments
    :param s:
    :param A:
    :param n:
    :return:
    '''
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


def first_round(classifier):
    class_ = -2
    while class_ != 9:

        num_stations = 1
        rate = 1  # np.random.uniform(0.5, 0.95)

        arrivals = get_ph()

        arrivals_norm = arrivals[3] / rate
        A = arrivals[1] * rate
        a = arrivals[0]
        moms_arrive = np.array(compute_first_n_moments(a, A, 10)).flatten()

        services_times = {}
        moms_ser = {}
        for station in range(num_stations):
            services = get_ph()
            rate = np.random.uniform(0.5, 0.95)
            ser_norm = services[3] * rate

            A = services[1] / rate
            a = services[0]

            moms_ser[station] = np.array(compute_first_n_moments(a, A, 10)).flatten()
            services_times[station] = ser_norm

        class_ = classifier.predict(np.append(moms_arrive[1:4], moms_ser[station][:4]).reshape(1, -1))[0]

    return services_times, rate, arrivals_norm, moms_arrive


def sec_round(classifier, moms_arrive):
    class_ = -2
    while class_ != 9:

        num_stations = 1

        services_times = {}
        moms_ser = {}
        for station in range(num_stations):
            services = get_ph()
            rate = np.random.uniform(0.5, 0.95)
            ser_norm = services[3] * rate

            A = services[1] / rate
            a = services[0]

            moms_ser[station] = np.array(compute_first_n_moments(a, A, 10)).flatten()
            services_times[station] = ser_norm

        class_ = classifier.predict(np.append(moms_arrive[1:4], moms_ser[station][:4]).reshape(1, -1))[0]

    return services_times, rate

for ind in range(500):
    try:
        services_times, rate, arrivals_norm, moms_arrive = first_round(classifier)
        services_times2, rate = sec_round(classifier, moms_arrive)
        services_times[1] = services_times2[0]

        model_name = str(np.random.randint(0,100000))+'_large_corrs.pkl'
        pkl.dump((arrivals_norm, services_times), open(os.path.join('/scratch/eliransc/ph_large_corrs', model_name), 'wb'))
        print('Succses#####################################')
    except:
        print('bad input')