import os
import numpy as np
from tqdm import tqdm
import pickle as pkl


cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]
path = '/scratch/eliransc/non_renewal/depart_0_train_long'

if not os.path.exists(path):
    os.mkdir(path)
files = os.listdir(path)
true_files = [file for file in files if 'multi' in file]
batch_size = 128

path_dump_data_depart_0 = '//scratch/eliransc/non_renewal/training_corrs/depart_0'
if not os.path.exists(path_dump_data_depart_0):
    os.mkdir(path_dump_data_depart_0)

num_batches = int(len(true_files)/batch_size)


for batch_num in tqdm(range(num_batches)):

    input_depart_0 = np.array([])
    output_depart_0 = np.array([])
    for batch_ind in range(batch_size):
        file_num = batch_num * batch_size + batch_ind
        try:
            inp, out = pkl.load(open(os.path.join(path, true_files[file_num]), 'rb'))
        except:
            print('Bad input')

        if batch_ind > 0:
            input_depart_0 = np.concatenate((inp.reshape(1, inp.shape[0]), input_depart_0), axis=0)
            output_depart_0 = np.concatenate((out.reshape(1, out.shape[0]), output_depart_0), axis=0)
        else:
            input_depart_0 = inp.reshape(1, inp.shape[0])
            output_depart_0 = out.reshape(1, out.shape[0])

    batch_name = 'train_long_large_scv_depart_0_from_'+cluster_name+'_batch_num_' + str(batch_num)+'.pkl'
    # print(os.path.join(path_dump_data_depart_0, batch_name), input_depart_0.shape, output_depart_0.shape)
    pkl.dump((input_depart_0, output_depart_0), open(os.path.join(path_dump_data_depart_0, batch_name), 'wb'))