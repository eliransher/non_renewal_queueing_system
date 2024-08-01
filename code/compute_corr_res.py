import pickle as pkl
import numpy as np
import os
from tqdm import tqdm

path  = '/scratch/eliransc/non_renewal/depart_1_scv1'
files = os.listdir(path)

corr_res_x = {}
corr_res_y = {}
for lower in np.linspace(-0.5, 0.4, 10):
    lower = round(lower, 1)
    corr_res_x[lower] = np.array([])
    corr_res_y[lower] = np.array([])

for file in tqdm(files):
    try:
        x,y = pkl.load(open(os.path.join(path, file), 'rb'))
        lower = round(float(file.split('_')[1]),1)
        if corr_res_x[lower].shape[0]> 0 :
            corr_res_x[lower] = np.concatenate((corr_res_x[lower], x.reshape(1, x.shape[0])), axis = 0)
            corr_res_y[lower] = np.concatenate((corr_res_y[lower], y.reshape(1, y.shape[0])), axis = 0)
        else:
            corr_res_x[lower] = x.reshape(1, x.shape[0])
            corr_res_y[lower] = y.reshape(1, y.shape[0])
    except:
        print('Ran out of input')
        os.remove(os.path.join(path, file))


pkl.dump((corr_res_x,corr_res_y), open('/scratch/eliransc/non_renewal/corr_res_depart_1_lower_corrs.pkl', 'wb'))