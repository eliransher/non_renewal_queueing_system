import os
import numpy as np
from tqdm import tqdm
import pickle as pkl



cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]
path = '/scratch/eliransc/non_renewal/steady_1_train_long'

if not os.path.exists(path):
    os.mkdir(path)
files = os.listdir(path)

true_files = [file for file in files if 'multi' in file]
# file_name_used = pkl.load( open('/scratch/eliransc/non_renewal/file_used_steady_1.pkl', 'rb'))
# true_files = [file for file in true_files if file not in file_name_used]

batch_size = 128

path_dump_data_depart_0 = '/scratch/eliransc/non_renewal/training_corrs/steady_1'
if not os.path.exists(path_dump_data_depart_0):
    os.mkdir(path_dump_data_depart_0)

num_batches = int(len(true_files)/batch_size)


for batch_num in tqdm(range(35,  num_batches)):

    input_depart_0 = np.array([])
    output_depart_0 = np.array([])
    for batch_ind in range(batch_size):
        file_num = batch_num * batch_size + batch_ind
        try:
            inp, out = pkl.load(open(os.path.join(path, true_files[file_num]), 'rb'))
        except:
            print('keep the same')
        # file_name_used.append(true_files[file_num])
        if batch_ind > 0:
            input_depart_0 = np.concatenate((input_depart_0, inp.reshape(1, inp.shape[0])), axis=0)
            output_depart_0 = np.concatenate((output_depart_0, out.reshape(1, out.shape[0])), axis=0)
        else:
            input_depart_0 = inp.reshape(1, inp.shape[0])
            output_depart_0 = out.reshape(1, out.shape[0])

    batch_name = 'train_long_large_scv_steady_1_from_'+cluster_name+'_batch_num_' + str(batch_num+len(os.listdir(path_dump_data_depart_0)))+'.pkl'

    pkl.dump((input_depart_0, output_depart_0), open(os.path.join(path_dump_data_depart_0, batch_name), 'wb'))
    # pkl.dump(file_name_used, open('/scratch/eliransc/non_renewal/file_used_steady_1.pkl', 'wb'))