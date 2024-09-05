import os
import numpy as np
from tqdm import tqdm
import pickle as pkl

cluster_name = os.listdir('/scratch/eliransc/cluster_name/')[0]

for folder in ['depart_0_low_util',
               'depart_0_scv1', ]:  # 'depart_0_train_long','depart_0_train_long3', 'depart_0_train_long2'
    path = os.path.join('/scratch/eliransc/non_renewal', folder)
    if not os.path.exists(path):
        os.mkdir(path)
    files = os.listdir(path)
    true_files = [file for file in files if 'multi' in file]
    batch_size = 128

    path_dump_data_depart_0 = '/scratch/eliransc/non_renewal/training_corrs/depart_0'  # '/scratch/eliransc/non_renewal/depart_0_from_narval/depart_0'
    if not os.path.exists(path_dump_data_depart_0):
        os.mkdir(path_dump_data_depart_0)

    num_batches = int(len(true_files) / batch_size)
    if folder == 'depart_0_low_util':
        init = 154
    else:
        init = 0
    for batch_num in tqdm(range(init, num_batches)):

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
        inds_good = (SCV_ser < 18) & (SCV_arrive < 18)
        inds_bad = ~inds_good
        temp_x = X[inds_good, :].copy()
        new_x = np.concatenate((X[inds_good, :], temp_x[:inds_bad.sum(), :]), axis=0)

        temp_y = y[inds_good, :].copy()
        new_y = np.concatenate((y[inds_good, :], temp_y[:inds_bad.sum(), :]), axis=0)

        batch_name = 'trial_' + folder + '_from_' + cluster_name + '_batch_num_' + str(batch_num) + '.pkl'
        print(os.path.join(path_dump_data_depart_0, batch_name), input_depart_0.shape, output_depart_0.shape)

        pkl.dump((new_x, new_y), open(os.path.join(path_dump_data_depart_0, batch_name), 'wb'))