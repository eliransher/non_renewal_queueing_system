import os
import numpy as np
from tqdm import tqdm
import pickle as pkl

## Train depart 0

file_name_used = []
cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]

batch_size = 128
## Train steady 1

file_name_used = []
cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]
path = '/scratch/eliransc/non_renewal/steady_1_low_util'
files = os.listdir(path)
true_files = [file for file in files if 'multi' in file]
num_batches = int(len(true_files)/batch_size)
true_files = true_files[:20000]
batch_size = 128

path_dump_data_steady_1 = '/scratch/eliransc/non_renewal/training_corrs/steady_1'
num_batches = int(len(true_files)/batch_size)
print(num_batches,len(true_files) )


for batch_num in tqdm(range(num_batches)):

    input_steady_1 = np.array([])
    output_steady_1 = np.array([])
    for batch_ind in range(batch_size):
        file_num = batch_num * batch_size + batch_ind
        try:
            inp, out = pkl.load(open(os.path.join(path, true_files[file_num]), 'rb'))
        except:
            print('keep the same')
        file_name_used.append(true_files[file_num])
        if batch_ind > 0:
            input_steady_1 = np.concatenate((input_steady_1, inp.reshape(1, inp.shape[0]) ), axis=0)
            output_steady_1 = np.concatenate((output_steady_1, out.reshape(1, out.shape[0]) ), axis=0)
        else:
            input_steady_1 = inp.reshape(1, inp.shape[0])
            output_steady_1 = out.reshape(1, out.shape[0])

    batch_name = 'low_utils_steady_1_from_'+cluster_name+'_batch_num_' + str(batch_num)+'.pkl'
    print(batch_name, input_steady_1.shape, output_steady_1.shape)
    pkl.dump((input_steady_1, output_steady_1), open(os.path.join(path_dump_data_steady_1, batch_name), 'wb'))

