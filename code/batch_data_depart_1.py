

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import os


# file_name_used = []
# cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]
# path = '/scratch/eliransc/non_renewal/depart_1_train_long3'
#
# files = os.listdir(path)
# true_files = [file for file in files if 'multi' in file]
# batch_size = 128
#
# path_dump_data_depart_1 = '/scratch/eliransc/non_renewal/training_corrs/depart_1'
# num_batches = int(len(true_files)/batch_size)
#
#
# for batch_num in tqdm(range(376,  num_batches)):
#
#     input_depart_1 = np.array([])
#     output_depart_1 = np.array([])
#     for batch_ind in range(batch_size):
#         file_num = batch_num * batch_size + batch_ind
#         try:
#             inp, out = pkl.load(open(os.path.join(path, true_files[file_num]), 'rb'))
#         except:
#             print('keep the same')
#         file_name_used.append(true_files[file_num])
#         if batch_ind > 0:
#             input_depart_1 = np.concatenate((input_depart_1, inp.reshape(1, inp.shape[0])), axis=0)
#             output_depart_1 = np.concatenate((output_depart_1, out.reshape(1, out.shape[0]), ), axis=0)
#         else:
#             input_depart_1 = inp.reshape(1, inp.shape[0])
#             output_depart_1 = out.reshape(1, out.shape[0])
#
#     batch_name = 'train_long2_depart_1_from_'+cluster_name+'_batch_num_' + str(batch_num)+'.pkl'
#
#     pkl.dump((input_depart_1, output_depart_1), open(os.path.join(path_dump_data_depart_1, batch_name), 'wb'))
#     # pkl.dump(file_name_used, open('/scratch/eliransc/non_renewal/file_used_depart_1.pkl', 'wb'))

# path = '/scratch/eliransc/non_renewal/training_corrs/depart_1'
# file_list = os.listdir(path)
# data_paths = [os.path.join(path, name) for name in file_list]
# len(data_paths)
#
# bad_path = '/scratch/eliransc/non_renewal/bad_batches'
# import shutil
#
# for patha in tqdm(data_paths):
#     # print(patha)
#     X, y = pkl.load(open(patha, 'rb'))
#
#     row_set = set(map(tuple, X))
#
#     # Check if there are duplicate rows
#     if len(row_set) < len(X):
#         print("There are identical rows in the array.")
#         dst = os.path.join(bad_path, patha.split('/')[-1])
#         # print(patha, dst)
#         shutil.move(patha, dst)
#     else:
#         pass



import os
import numpy as np
from tqdm import tqdm
import pickle as pkl

cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]

for folder in ['new_depart_1']:
    path = os.path.join('/scratch/eliransc/non_renewal', folder)
    if not os.path.exists(path):
        os.mkdir(path)
    files = os.listdir(path)
    true_files = [file for file in files if 'multi' in file]
    batch_size = 128

    path_dump_data_depart_0 = '/scratch/eliransc/non_renewal/training_corrs/new_depart_1'  # '/scratch/eliransc/non_renewal/depart_0_from_narval/depart_0'
    if not os.path.exists(path_dump_data_depart_0):
        os.mkdir(path_dump_data_depart_0)

    num_batches = int(len(true_files) / batch_size)

    for batch_num in tqdm(range(num_batches)):

        input_depart_0 = np.array([])
        output_depart_0 = np.array([])
        for batch_ind in range(batch_size):
            file_num = batch_num * batch_size + batch_ind
            try:
                inp, out = pkl.load(open(os.path.join(path, true_files[file_num]), 'rb'))
            except:
                try:
                    inp, out = pkl.load(open(os.path.join(path, true_files[file_num - 1]), 'rb'))
                except:
                    inp, out = pkl.load(open(os.path.join(path, true_files[0]), 'rb'))

            if batch_ind > 0:
                input_depart_0 = np.concatenate((inp.reshape(1, inp.shape[0]), input_depart_0), axis=0)
                output_depart_0 = np.concatenate((out.reshape(1, out.shape[0]), output_depart_0), axis=0)
            else:
                input_depart_0 = inp.reshape(1, inp.shape[0])
                output_depart_0 = out.reshape(1, out.shape[0])

        X = input_depart_0
        y = output_depart_0
        SCV_ser = (np.exp(X[:, -9]) - np.exp(X[:, -10]) ** 2) / np.exp(X[:, -10]) ** 2
        SCV_arrive = (np.exp(X[:, 1]) - np.exp(X[:, 0]) ** 2) / np.exp(X[:, 0]) ** 2
        print(SCV_ser.max(), SCV_arrive.max())

        inds_good = (SCV_ser < 15) & (SCV_arrive < 15)
        inds_bad = ~inds_good
        temp_x = X[inds_good, :].copy()
        new_x = np.concatenate((X[inds_good, :], temp_x[:inds_bad.sum(), :]), axis=0)

        temp_y = y[inds_good, :].copy()
        new_y = np.concatenate((y[inds_good, :], temp_y[:inds_bad.sum(), :]), axis=0)

        batch_name = 'new_' + folder + '_from_' + cluster_name + '_batch_num_' + str(batch_num) + '.pkl'
        print(os.path.join(path_dump_data_depart_0, batch_name), input_depart_0.shape, output_depart_0.shape)

        pkl.dump((new_x, new_y), open(os.path.join(path_dump_data_depart_0, batch_name), 'wb'))