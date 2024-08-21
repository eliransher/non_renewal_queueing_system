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
m = nn.Softmax(dim=1)
import matplotlib.pyplot as plt
import time


def pare_per_batch(preds, labels, utils, prectile):
    errors = []
    for ind in range(preds.shape[0]):
        errors.append(100 * PARE(preds[ind, :], labels[ind, :], utils[ind], prectile))
    return np.array(errors)


def PARE(preds, labels, util, prectile):
    flag_preds = True
    flag_true = True
    ind_preds = False
    ind_true = False
    for ind in range(preds.shape[0]):

        if flag_preds:
            if 1 - util + torch.sum(preds[:ind]) > prectile:
                ind_preds = ind + 1
                flag_preds = False
        if flag_true:
            if torch.sum(labels[:ind]) > prectile:
                ind_true = ind
                flag_true = False
    if ind_preds and ind_true:
        if ind_true > 0:
            return abs((ind_true - ind_preds) / ind_true)
            # relative_error.append(abs((ind_true-ind_preds)/ind_true))
        else:
            return 0.0001


def queue_loss(predictions, targes, utilization):
    normalizing_const = utilization

    predictions = m(predictions)
    predictions = predictions * normalizing_const

    return ((torch.pow(torch.abs(predictions - targes[:, 1:]), 1)).sum(axis=1) +
            torch.max(torch.pow(torch.abs(predictions - targes[:, 1:]), 1), 1)[0]).sum()


def valid(dset_val, model, num_ser_moms):
    loss = 0
    for batch in dset_val:
        X_valid, y_valid = batch
        X_valid = X_valid.float()
        X_valid = X_valid.reshape(X_valid.shape[1], X_valid.shape[2])
        y_valid = y_valid.float()
        y_valid = y_valid.reshape(y_valid.shape[1], y_valid.shape[2])
        X_valid = X_valid.to(device)
        y_valid = y_valid.to(device)
        utils = ((1 / torch.exp(X_valid[:, 0])) * torch.exp(X_valid[:, -num_ser_moms])).reshape(-1, 1)
        loss += queue_loss(model(X_valid), y_valid, utils)
    return loss / len(dset_val)


def check_loss_increasing(loss_list, n_last_steps=10, failure_rate=0.45):
    try:
        counter = 0
        curr_len = len(loss_list)
        if curr_len < n_last_steps:
            n_last_steps = curr_len

        inds_arr = np.linspace(n_last_steps - 1, 1, n_last_steps - 1).astype(int)
        for ind in inds_arr:
            if loss_list[-ind] > loss_list[-ind - 1]:
                counter += 1

        print(counter, n_last_steps)
        if counter / n_last_steps > failure_rate:
            return True

        else:
            return False
    except:
        return False


def compute_sum_error_large_corrs(valid_test_x, valid_test_y, model, num_ser_moms):

    with torch.no_grad():
        X_valid = valid_test_x.float()
        y_valid = valid_test_y.float()

        X_valid = X_valid.to(device)
        y_valid = y_valid.to(device)

        targes = y_valid

        utils = ((1 / torch.exp(X_valid[:, 0])) * torch.exp(X_valid[:, -num_ser_moms])).reshape(-1, 1)
        normalizing_const = utils
        predictions = model(X_valid)
        predictions = m(predictions)
        predictions = predictions * normalizing_const

        error = (torch.pow(torch.abs(predictions - targes[:, 1:]), 1)).sum(axis=1)

        return error.mean().item()


def compute_sum_error(valid_dl, model, num_ser_moms, return_vector, max_err=0.05, display_bad_images=False):
    with torch.no_grad():
        errors = []

        for batch in valid_dl:
            X_valid, y_valid = batch
            X_valid = X_valid.float()
            X_valid = X_valid.reshape(X_valid.shape[1], X_valid.shape[2])
            y_valid = y_valid.float()
            y_valid = y_valid.reshape(y_valid.shape[1], y_valid.shape[2])
            X_valid = X_valid.to(device)
            y_valid = y_valid.to(device)

            targes = y_valid

            utils = ((1 / torch.exp(X_valid[:, 0])) * torch.exp(X_valid[:, -num_ser_moms])).reshape(-1, 1)
            normalizing_const = utils
            predictions = model(X_valid)
            predictions = m(predictions)
            predictions = predictions * normalizing_const

            error = (torch.pow(torch.abs(predictions - targes[:, 1:]), 1)).sum(axis=1)
            errors.append(error.mean())

    return torch.tensor(errors).mean()


class my_Dataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms=5, num_ser_moms=5):
        self.data_paths = data_paths
        self.max_lag = max_lag
        self.max_power_1 = max_power_1
        self.max_power_2 = max_power_2
        self.num_arrival_moms = num_arrival_moms
        self.num_ser_moms = num_ser_moms
        self.df = df

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x, y = pkl.load(open(self.data_paths[index], 'rb'))

        x1 = x[:, :self.num_arrival_moms]
        x2 = x[:,
             self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (self.df['mom_2'] <= self.max_power_2),
             :].index + 10]
        x3 = x[:, -10: -10 + self.num_ser_moms]
        x = np.concatenate((x1, x2, x3), axis=1)

        inputs = torch.from_numpy(x)
        y = torch.from_numpy(y[:, :1500])

        return x, y


class my_Dataset_set2(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms=5, num_ser_moms=5):
        self.data_paths = data_paths
        self.max_lag = max_lag
        self.max_power_1 = max_power_1
        self.max_power_2 = max_power_2
        self.num_arrival_moms = num_arrival_moms
        self.num_ser_moms = num_ser_moms
        self.df = df

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x, y = pkl.load(open(self.data_paths[index], 'rb'))

        x1 = x[:, :self.num_arrival_moms]
        x2 = x[:,
             self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (self.df['mom_2'] <= self.max_power_2),
             :].index + 10]
        x3 = x[:, -5:]
        x = np.concatenate((x1, x2, x3), axis=1)

        inputs = torch.from_numpy(x)
        y = torch.from_numpy(y[:, :1500])

        return x, y


class Net(nn.Module):

    def __init__(self, input_size, output_size, type_archi):
        self.type_archi = type_archi
        super().__init__()
        if type_archi == 6:
            self.fc1 = nn.Linear(input_size, 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 200)
            self.fc4 = nn.Linear(200, 350)
            self.fc5 = nn.Linear(350, 600)
            self.fc6 = nn.Linear(600, output_size)
        else:
            self.fc1 = nn.Linear(input_size, 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 100)
            self.fc4 = nn.Linear(100, 200)
            self.fc5 = nn.Linear(200, 350)
            self.fc6 = nn.Linear(350, 600)
            self.fc7 = nn.Linear(600, output_size)

    def forward(self, x):

        if self.type_archi == 6:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = self.fc6(x)
            return x
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
            x = self.fc7(x)
            return x

def main():


    path = '/scratch/eliransc/non_renewal/training_corrs/steady_1'
    file_list = os.listdir(path)

    thresh = np.random.choice([0.2, 0.3], p=[0.5, 0.5])
    if True:
        train_files = []
        for file in file_list:
            if file.split('_')[0] == 'batch':
                if np.abs(float(file.split('_')[-1][:-4])) > thresh:
                    train_files.append(file)
            else:
                train_files.append(file)
        data_paths = [os.path.join(path, name) for name in train_files]
    else:
        data_paths = [os.path.join(path, name) for name in file_list]
    len(data_paths)


    try:
        cur_time = int(1000*time.time())
        seed = cur_time  # + len(os.listdir(data_path)) +
        np.random.seed(int(seed/1000))
        print(seed)
    except:
        cur_time = int(time.time())
        np.random.seed(cur_time)
        print(seed)

    path = '/scratch/eliransc/non_renewal/training_corrs/steady_1'
    file_list = os.listdir(path)
    data_paths = [os.path.join(path, name) for name in file_list]
    len(data_paths)

    try:
        cur_time = int(1000*time.time())
        seed = cur_time  # + len(os.listdir(data_path)) +
        np.random.seed(int(seed/1000))
        print(seed)
    except:
        cur_time = int(time.time())
        np.random.seed(cur_time)
        print(seed)


    num_arrival_moms = 5
    num_ser_moms = 5
    max_lag = 2 # np.random.randint(0, 6)
    max_power_1 = 2 # np.random.randint(1, 6)
    max_power_2 = 2 # max_power_1

    df = pd.DataFrame([])

    for corr_leg in range(1, 6):
        for mom_1 in range(1, 6):
            for mom_2 in range(1, 6):
                curr_ind = df.shape[0]
                df.loc[curr_ind, 'lag'] = corr_leg
                df.loc[curr_ind, 'mom_1'] = mom_1
                df.loc[curr_ind, 'mom_2'] = mom_2
                df.loc[curr_ind, 'index'] = curr_ind + 10



    dataset = my_Dataset(data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms, num_ser_moms)
    batch_size = 1
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)

    # get first sample and unpack
    first_data = dataset[1]
    features, labels = first_data


    valid_loader = {}
    test_paths = '/scratch/eliransc/non_renewal/test_corrs/steady_1'
    for folder in os.listdir(test_paths):
        path_test = os.path.join(test_paths, folder)
        files_valid = os.listdir(path_test)
        valid_path = [os.path.join(path_test, name) for name in files_valid]
        dataset_valid = my_Dataset(valid_path, df, max_lag, max_power_1, max_power_2, num_arrival_moms, num_ser_moms)

        valid_loader[folder] = DataLoader(dataset=dataset_valid,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1)

    archi_type = np.random.randint(6, 8)
    input_size = features.shape[1]
    output_size = labels.shape[1] - 1
    net = Net(input_size, output_size, archi_type).to(device)
    weight_decay = 5
    curr_lr = np.random.choice([0.001,0.01, 0.05], p=[0.2, 0.4,0.4])
    EPOCHS = 300
    lr_second = 0.99
    lr_first = 0.75


    # Construct dataset

    optimizer = optim.Adam(net.parameters(), lr=curr_lr,
                           weight_decay=(1 / 10 ** weight_decay))  # paramters is everything adjustable in model

    loss_list = []
    valid_list = {}
    compute_sum_error_list = {}
    for folder in os.listdir(test_paths):
        valid_list[folder] = []
        compute_sum_error_list[folder] = []

    model_path = '/scratch/eliransc/non_renewal/models'
    model_results_path = '/scratch/eliransc/non_renewal/models_results'
    model_num = np.random.randint(0, 1000000)
    file_name_model = 'model_' + str(model_num) +'threshdata_'+ str(thresh)+ '_wd_' + str(weight_decay) + '_lr_' + str(curr_lr) + '_pow1_' + str(
        max_power_1) + '_pow2_' + str(max_power_2) + '_maxlag_' + str(max_lag) + '_layer_' + str(archi_type) + '.pkl'
    file_name_model_result = 'mmodelresults_' + str(model_num) +'threshdata_'+ str(thresh)+ '_wd_' + str(weight_decay) + '_lr_' + str(curr_lr) + '_pow1_' + str(
        max_power_1) + '_pow2_' + str(max_power_2) + '_maxlag_' + str(max_lag) + '_layer_' + str(archi_type) + '.pkl'

    file_name_df_tot = 'df_tot_' + str(model_num) + 'threshdata_' + str(thresh) + '_wd_' + str(
        weight_decay) + '_lr_' + str(curr_lr) + '_pow1_' + str(
        max_power_1) + '_pow2_' + str(max_power_2) + '_maxlag_' + str(max_lag) + '_layer_' + str(archi_type) + '.pkl'

    file_name_df_tot1 = 'df_tot1_' + str(model_num) + 'threshdata_' + str(thresh) + '_wd_' + str(
        weight_decay) + '_lr_' + str(curr_lr) + '_pow1_' + str(
        max_power_1) + '_pow2_' + str(max_power_2) + '_maxlag_' + str(max_lag) + '_layer_' + str(archi_type) + '.pkl'


    print(file_name_model_result)
    num_probs_presenet = 20
    print('start training')
    for epoch in tqdm(range(EPOCHS)):

        cur_time = int(time.time())
        np.random.seed(cur_time)

        for i, (X, y) in enumerate(train_loader):

            X = X.float()
            X = X.reshape(X.shape[1], X.shape[2])
            y = y.float()
            y = y.reshape(y.shape[1], y.shape[2])
            X = X.to(device)
            y = y.to(device)

            if torch.sum(torch.isinf(X)).item() == 0:

                net.zero_grad()
                output = net(X)
                loss = queue_loss(output, y, ((1 / torch.exp(X[:, 0])) * torch.exp(X[:, -num_ser_moms])).reshape(-1,1))
                loss.backward()
                optimizer.step()
                net.zero_grad()



        print('Compute validation')
        loss_list.append(loss.item())
        for folder in os.listdir(test_paths):
            valid_list[folder].append(valid(valid_loader[folder], net, num_ser_moms).item())
            compute_sum_error_list[folder].append(
                compute_sum_error(valid_loader[folder], net, num_ser_moms, False).item())

        if len(loss_list) > 5:
            if check_loss_increasing(valid_list):
                curr_lr = curr_lr * lr_first
                optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                print(curr_lr)
            else:
                curr_lr = curr_lr * lr_second
                optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                print(curr_lr)

        for folder in os.listdir(test_paths):
            print('SAE: ', folder, compute_sum_error_list[folder][-1])


        torch.save(net.state_dict(), os.path.join(model_path, file_name_model))

        if (epoch > 10) & ( epoch %10 == 0):
            m = nn.Softmax(dim=1)

            df_tot = pd.DataFrame([])

            for folder in tqdm(os.listdir(test_paths)):
                with torch.no_grad():

                    errors = []

                    for batch in tqdm(valid_loader[folder]):
                        X_valid, y_valid = batch
                        X_valid = X_valid.float()
                        X_valid = X_valid.reshape(X_valid.shape[1], X_valid.shape[2])
                        y_valid = y_valid.float()
                        y_valid = y_valid.reshape(y_valid.shape[1], y_valid.shape[2])
                        X_valid = X_valid.to(device)
                        y_valid = y_valid.to(device)

                        targes = y_valid

                        utils = ((1 / torch.exp(X_valid[:, 0])) * torch.exp(X_valid[:, -num_ser_moms])).reshape(-1, 1)
                        normalizing_const = utils
                        predictions = net(X_valid)
                        predictions = m(predictions)
                        predictions = predictions * normalizing_const

                        error = (torch.pow(torch.abs(predictions - targes[:, 1:]), 1)).sum(axis=1)
                        # print(error)
                        errors.append(error.mean())
                        SCV_ser = (torch.exp(X_valid[:, -4]) - torch.exp(X_valid[:, -5]) ** 2) / torch.exp(
                            X_valid[:, -5]) ** 2
                        SCV_arrive = (torch.exp(X_valid[:, 1]) - torch.exp(X_valid[:, 0]) ** 2) / torch.exp(
                            X_valid[:, 0]) ** 2
                        # print(SCV_ser.max(), SCV_arrive.max(), SCV_ser.min(), SCV_arrive.min())
                        utilization = 1 - y_valid[:, 0]
                        rhos = X_valid[:, 5]

                        df_res = pd.DataFrame([])
                        df_res['SCV_ser'] = np.array(SCV_ser.to('cpu'))
                        df_res['SCV_arrive'] = np.array(SCV_arrive.to('cpu'))
                        df_res['utilization'] = np.array(utilization.to('cpu'))
                        df_res['rhos'] = np.array(rhos.to('cpu'))
                        df_res['SAE'] = np.array(error.to('cpu'))
                        for perentile in [0.25, 0.5, 0.75, 0.9, 0.99, 0.999]:
                            res = pare_per_batch(predictions, targes, utils, perentile)
                            df_res['PARE_' + str(perentile)] = res

                        mean_preds = (np.arange(1, predictions.shape[1] + 1) * predictions.cpu().numpy()[:, :]).sum(
                            axis=1)
                        mean_labels = (np.arange(1, targes.shape[1]) * targes.cpu().numpy()[:, 1:]).sum(axis=1)
                        REM = 100 * ((np.abs(mean_preds - mean_labels)) / mean_labels)
                        df_res['REM'] = REM

                        df_tot = pd.concat([df_tot, df_res], axis=0)

                        pkl.dump(df_tot, open(os.path.join(model_results_path, file_name_df_tot), 'wb'))

                    print('SAE test 1')
                    print(df_tot.loc[(df_tot['SCV_ser'] < 1), 'SAE'].mean())
                    print(df_tot.loc[(df_tot['SCV_ser'] > 1), 'SAE'].mean())

                    print('PARE 99 test 1')
                    print(df_tot.loc[(df_tot['SCV_ser'] < 1), 'PARE_0.9'].mean())
                    print(df_tot.loc[(df_tot['SCV_ser'] > 1), 'PARE_0.9'].mean())



        test2_path = '/scratch/eliransc/steady_1'
        files_set2 = os.listdir(test2_path)
        data_paths_set2 = [os.path.join(test2_path, file) for file in files_set2]
        dataset_set2 = my_Dataset_set2(data_paths_set2, df, max_lag, max_power_1, max_power_2, num_arrival_moms,
                                       num_ser_moms)
        batch_size = 1
        testset2_loader = DataLoader(dataset=dataset_set2,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=1)

        with torch.no_grad():
            df_tot1 = pd.DataFrame([])
            errors = []
            m = nn.Softmax(dim=1)
            for batch in tqdm(testset2_loader):
                X_valid, y_valid = batch
                X_valid = X_valid.float()
                X_valid = X_valid.reshape(X_valid.shape[1], X_valid.shape[2])
                y_valid = y_valid.float()
                y_valid = y_valid.reshape(y_valid.shape[1], y_valid.shape[2])
                X_valid = X_valid.to(device)
                y_valid = y_valid.to(device)

                targes = y_valid

                utils = ((1 / torch.exp(X_valid[:, 0])) * torch.exp(X_valid[:, -num_ser_moms])).reshape(-1, 1)
                normalizing_const = utils
                predictions = net(X_valid)
                predictions = m(predictions)
                predictions = predictions * normalizing_const

                error = (torch.pow(torch.abs(predictions - targes[:, 1:]), 1)).sum(axis=1)
                # print(error)
                errors.append(error.mean())
                SCV_ser = (torch.exp(X_valid[:, -4]) - torch.exp(X_valid[:, -5]) ** 2) / torch.exp(X_valid[:, -5]) ** 2
                SCV_arrive = (torch.exp(X_valid[:, 1]) - torch.exp(X_valid[:, 0]) ** 2) / torch.exp(X_valid[:, 0]) ** 2
                # print(SCV_ser.max(), SCV_arrive.max(), SCV_ser.min(), SCV_arrive.min())
                utilization = 1 - y_valid[:, 0]
                rhos = X_valid[:, 5]

                df_res = pd.DataFrame([])
                df_res['SCV_ser'] = np.array(SCV_ser.to('cpu'))
                df_res['SCV_arrive'] = np.array(SCV_arrive.to('cpu'))
                df_res['utilization'] = np.array(utilization.to('cpu'))
                df_res['rhos'] = np.array(rhos.to('cpu'))
                df_res['SAE'] = np.array(error.to('cpu'))

                df_tot1 = pd.concat([df_tot1, df_res], axis=0)
                df_tot1['rho_group'] = pd.qcut(df_tot1['utilization'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                for perentile in [0.25, 0.5, 0.75, 0.9, 0.99, 0.999]:
                    res = pare_per_batch(predictions, targes, utils, perentile)
                    df_tot1['PARE_' + str(perentile)] = res

                mean_preds = (np.arange(1, predictions.shape[1] + 1) * predictions.cpu().numpy()[:, :]).sum(axis=1)
                mean_labels = (np.arange(1, targes.shape[1]) * targes.cpu().numpy()[:, 1:]).sum(axis=1)
                REM = 100 * ((np.abs(mean_preds - mean_labels)) / mean_labels)
                df_tot1['REM'] = REM

                pkl.dump(df_tot1, open(os.path.join(model_results_path, file_name_df_tot1), 'wb'))



                print('SAE test 2')
                print(df_tot1.loc[(df_tot1['SCV_ser'] < 1), 'SAE'].mean())
                print(df_tot1.loc[(df_tot1['SCV_ser'] > 1), 'SAE'].mean())

                print('PARE 99 test 2')
                print(df_tot1.loc[(df_tot1['SCV_ser'] < 1), 'PARE_0.9'].mean())
                print(df_tot1.loc[(df_tot1['SCV_ser'] > 1), 'PARE_0.9'].mean())

        df_tot = pkl.load(open(os.path.join(model_results_path, file_name_df_tot), 'rb'))
        df_tot1 = pkl.load(open(os.path.join(model_results_path, file_name_df_tot1), 'rb'))

        pkl.dump((df_tot, df_tot1, compute_sum_error_list, valid_list, max_lag, max_power_1, max_power_2, epoch),
                 open(os.path.join(model_results_path, file_name_model_result), 'wb'))
        print(os.path.join(model_results_path, file_name_model_result))
if __name__ == "__main__":
    main()