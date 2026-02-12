import os
import sys

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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device



def scv_partion(df):
    # --- binning ---
    def bin_scv(x):
        if pd.isna(x): return np.nan
        return "<3" if x < 3 else ">3"

    def bin_mom1ratio(x):
        if pd.isna(x): return np.nan
        return "<0.25" if x < 0.25 else ">0.25"

    def bin_rho(x):
        if pd.isna(x): return np.nan
        if x < -0.25:
            return "<-0.25"
        elif x < 0:
            return "(-0.25,0)"
        elif x < 0.25:
            return "(0,0.25)"
        elif x < 1:
            return "(0.25,1)"
        else:
            return ">=1"  # outside requested bins

    df["SCV1_bin"] = df["SCV1"].apply(bin_scv)
    df["SCV2_bin"] = df["SCV2"].apply(bin_scv)
    df["rho1_bin"] = df["rho1"].apply(bin_rho)
    df["rho2_bin"] = df["rho2"].apply(bin_rho)
    df["mom1ratio_bin"] = df["mom1ratio"].apply(bin_mom1ratio)

    err_cols = ["err2", "err3", "err4", "err5"]

    # ----------------------------
    # 1) Split by combined (rho1, rho2, mom1ratio)
    # ----------------------------
    group_rhos = ["rho1_bin", "rho2_bin", "mom1ratio_bin"]

    summary_rhos = (
        df.groupby(group_rhos, dropna=False)[err_cols]
        .agg(["count", "mean"])
    )
    summary_rhos.columns = [f"{c}_{stat}" for c, stat in summary_rhos.columns]
    summary_rhos = summary_rhos.reset_index()

    # Include empty groups (cartesian product of requested bins)
    rho_bins = ["<-0.25", "(-0.25,0)", "(0,0.25)", "(0.25,1)"]
    mom_bins = ["<0.25", ">0.25"]

    full_idx_rhos = pd.MultiIndex.from_product([rho_bins, rho_bins, mom_bins], names=group_rhos)
    summary_rhos_full = (
        df[df["rho1_bin"].isin(rho_bins) & df["rho2_bin"].isin(rho_bins)]
        .groupby(group_rhos, dropna=False)[err_cols]
        .agg(["count", "mean"])
    )
    summary_rhos_full.columns = [f"{c}_{stat}" for c, stat in summary_rhos_full.columns]
    summary_rhos_full = summary_rhos_full.reindex(full_idx_rhos).reset_index()
    summary_rhos_full[[f"{c}_count" for c in err_cols]] = summary_rhos_full[[f"{c}_count" for c in err_cols]].fillna(
        0).astype(int)

    # ----------------------------
    # 2) Split by combined (SCV1, SCV2, mom1ratio)
    # ----------------------------
    group_scvs = ["SCV1_bin", "SCV2_bin", "mom1ratio_bin"]

    summary_scvs = (
        df.groupby(group_scvs, dropna=False)[err_cols]
        .agg(["count", "mean"])
    )
    summary_scvs.columns = [f"{c}_{stat}" for c, stat in summary_scvs.columns]
    summary_scvs = summary_scvs.reset_index()

    # Include empty groups
    scv_bins = ["<3", ">3"]
    full_idx_scvs = pd.MultiIndex.from_product([scv_bins, scv_bins, mom_bins], names=group_scvs)

    summary_scvs_full = (
        df.groupby(group_scvs, dropna=False)[err_cols]
        .agg(["count", "mean"])
    )
    summary_scvs_full.columns = [f"{c}_{stat}" for c, stat in summary_scvs_full.columns]
    summary_scvs_full = summary_scvs_full.reindex(full_idx_scvs).reset_index()
    summary_scvs_full[[f"{c}_count" for c in err_cols]] = summary_scvs_full[[f"{c}_count" for c in err_cols]].fillna(
        0).astype(int)

    # ---- outputs ----
    # summary_rhos_full: includes empty rho1/rho2/mom1ratio groups
    # summary_scvs_full: includes empty SCV1/SCV2/mom1ratio groups

    return summary_scvs_full, summary_rhos_full


def check_test(loader_valid, net_moms):
    all_errs = []
    for X1, y1 in loader_valid:
        X1 = X1.float()
        y1 = y1.float()
        X1 = X1.to(device)
        y1 = y1.to(device)
        X1 = X1[:, 0, :]
        y1 = y1[:, 0, :]
        SCV_1 = (torch.exp(X1[:, 1]) - torch.exp(X1[:, 0]) ** 2) / torch.exp(X1[:, 0]) ** 2
        SCV_2 = (torch.exp(X1[:, 14]) - torch.exp(X1[:, 13]) ** 2) / torch.exp(X1[:, 13]) ** 2
        mom1_ratio = torch.exp(X1[:, 0])
        rho_1 = X1[:, 5]
        rho_2 = X1[:, 18]

        preds = net_moms(X1)
        curr_errs = 100 * torch.abs((torch.exp(preds[:, :]) - torch.exp(y1[:, :])) / torch.exp(y1[:, :])).mean(axis=0)
        curr_torch = 100 * torch.abs((torch.exp(preds[:, :]) - torch.exp(y1[:, :])) / torch.exp(y1[:, :]))

        curr_torch = torch.concatenate((curr_torch, SCV_1.reshape(-1, 1)), axis=1)
        curr_torch = torch.concatenate((curr_torch, SCV_2.reshape(-1, 1)), axis=1)

        curr_torch = torch.concatenate((curr_torch, rho_1.reshape(-1, 1)), axis=1)
        curr_torch = torch.concatenate((curr_torch, rho_2.reshape(-1, 1)), axis=1)

        curr_torch = torch.concatenate((curr_torch, mom1_ratio.reshape(-1, 1)), axis=1)

        all_errs.append(curr_torch)

    tot_mom_res = torch.vstack(all_errs)

    with torch.no_grad():
        sumres = pd.DataFrame(tot_mom_res.cpu(),
                              columns=['err2', 'err3', 'err4', 'err5', 'SCV1', 'SCV2', 'rho1', 'rho2', 'mom1ratio'])

    return sumres





def valid(dataset_val, model):
    loss = 0

    for X, y in loader_valid:
        X = X.float()
        y = y.float()
        X = X.to(device)
        y = y.to(device)
        X = X[:, 0, :]
        y = y[:, 0, :]
        loss += depart_loss(model(X), y)
    return loss / len(dataset_val)


def valid_rel_err(dataset_val, model):
    for X, y in loader_valid:
        X = X.float()
        y = y.float()
        X = X.to(device)
        y = y.to(device)
        X = X[:, 0, :]
        y = y[:, 0, :]
        loss += depart_loss(model(X), y)
    return loss / len(dataset_val)


def check_loss_increasing(loss_list, n_last_steps=10, failure_rate=0.45):
    counter = 0
    curr_len = len(loss_list)
    if curr_len < n_last_steps:
        n_last_steps = curr_len

    inds_arr = np.linspace(n_last_steps - 1, 1, n_last_steps - 1).astype(int)
    for ind in inds_arr:
        if loss_list[-ind] > loss_list[-ind - 1]:
            counter += 1

    # print(counter, n_last_steps)
    if counter / n_last_steps > failure_rate:
        return True

    else:
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





def depart_loss_correlation(preds, target):
    weights_corr = torch.flip(torch.arange(1, num_moms_corrs + 1), dims=(0,))
    weights_corr = weights_corr.to(device)
    corr_loss_ = weights_corr * torch.abs((preds[:, :] - target[:, :]))
    corr_loss = corr_loss_[corr_loss_ < 100000].mean()

    return corr_loss


def depart_loss(preds, target, num_moms_depart=5):
    weights = torch.flip(torch.arange(1, num_moms_depart), dims=(0,))
    weights = weights.to(device)
    # loss_arrive = torch.abs((preds[:,:num_moms_depart]-target[:,:num_moms_depart])/target[:,:num_moms_depart]) #  weights*
    loss_arrive = (preds[:, :num_moms_depart] - target[:, :num_moms_depart]) ** 2
    # loss_arrive = loss_arrive[loss_arrive<100000]
    moms_loss_arrive = loss_arrive.mean()

    return moms_loss_arrive


class my_Dataset_moms(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms=5, max_lag_y=2, num_moms_y=5,
                 max_power_1_y=2, max_power_2_y=2):
        self.data_paths = data_paths
        self.max_lag = max_lag
        self.max_power_1 = max_power_1
        self.max_power_2 = max_power_2
        self.num_arrival_moms = num_arrival_moms
        self.df = df

        self.num_moms_y = num_moms_y
        self.max_power_1_y = max_power_1_y
        self.max_power_2_y = max_power_2_y
        self.max_lag_y = max_lag_y

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x, y = pkl.load(open(self.data_paths[index], 'rb'))

        y = y.reshape(1, -1)

        x = x.reshape(1, -1)

        x1 = np.log(x[:, :self.num_arrival_moms])
        x2 = x[:,
             self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (self.df['mom_2'] <= self.max_power_2),
             :].index + 10]
        x3 = np.log(x[:, 135:135 + self.num_arrival_moms])
        x4 = x[:, 145 + self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (
                    self.df['mom_2'] <= self.max_power_2), :].index]
        x = np.concatenate((x1, x2, x3, x4), axis=1)
        x = torch.from_numpy(x)

        y1 = y[:, 1:self.num_moms_y]
        # y2 = y[:,df.loc[(df['lag']<=self.max_lag_y)&(df['mom_1']<=self.max_power_1_y)&(df['mom_2']<=self.max_power_2_y),:].index+10]
        # y = np.concatenate((y1,y2), axis = 1)
        y = torch.log(torch.from_numpy(y1))

        return x, y


class my_Dataset_corrs(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms=5, max_lag_y=2, num_moms_y=5,
                 max_power_1_y=2, max_power_2_y=2):
        self.data_paths = data_paths
        self.max_lag = max_lag
        self.max_power_1 = max_power_1
        self.max_power_2 = max_power_2
        self.num_arrival_moms = num_arrival_moms
        self.df = df

        self.num_moms_y = num_moms_y
        self.max_power_1_y = max_power_1_y
        self.max_power_2_y = max_power_2_y
        self.max_lag_y = max_lag_y

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x, y = pkl.load(open(self.data_paths[index], 'rb'))

        y = y.reshape(1, -1)

        x = x.reshape(1, -1)

        x1 = np.log(x[:, :self.num_arrival_moms])
        x2 = x[:, self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (
                    self.df['mom_2'] <= self.max_power_2), :].index + 10]
        x3 = np.log(x[:, 135:135 + self.num_arrival_moms])
        x4 = x[:, 145 + self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (
                    self.df['mom_2'] <= self.max_power_2), :].index]
        x = np.concatenate((x1, x2, x3, x4), axis=1)
        x = torch.from_numpy(x)

        # y1 = y[:, :self.num_moms_y]
        y2 = y[:, self.df.loc[(self.df['lag'] <= self.max_lag_y) & (df['mom_1'] <= self.max_power_1_y) & (
                    self.df['mom_2'] <= self.max_power_2_y), :].index + 10]
        # y = np.concatenate((y1,y2), axis = 1)
        y = torch.log(torch.from_numpy(y2))

        return x, y

def main():

    if sys.platform == 'linux':
        path = '/scratch/eliransc/MAP/training/merge_1'

        path_valid = '/scratch/eliransc/MAP/valid/merge_valid'
    else:

        path_valid = r'C:\Users\Eshel\workspace\data\merge_data\merge_valid'
        path = r'C:\\Users\\Eshel\\workspace\\data\\merge_data\\merge_1'

    num_arrival_moms = 5
    max_lag = 2
    max_power_1 = 2
    max_power_2 = 2

    df = pd.DataFrame([])

    for corr_leg in range(1, 6):
        for mom_1 in range(1, 6):
            for mom_2 in range(1, 6):
                curr_ind = df.shape[0]
                df.loc[curr_ind, 'lag'] = corr_leg
                df.loc[curr_ind, 'mom_1'] = mom_1
                df.loc[curr_ind, 'mom_2'] = mom_2
                df.loc[curr_ind, 'index'] = curr_ind + 10

    files = os.listdir(path)

    data_paths = [os.path.join(path, name) for name in files]

    dataset = my_Dataset_moms(data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms)

    files_valid = os.listdir(path_valid)
    data_paths_valid = [os.path.join(path_valid, name) for name in files_valid]

    dataset_valid = my_Dataset_moms(data_paths_valid, df, max_lag, max_power_1, max_power_2, num_arrival_moms)

    dataset_corrs = my_Dataset_corrs(data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms)

    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=128, shuffle=True)

    import torch
    import torch.nn as nn

    if np.random.rand() < 0.5:
        nn_archi = 1
    else:
        nn_archi = 2

    if nn_archi == 1:

        class Net(nn.Module):

            def __init__(self, input_size, output_size):
                super().__init__()

                self.fc1 = nn.Linear(input_size, 50)
                self.fc2 = nn.Linear(50, 70)
                self.fc3 = nn.Linear(70, 50)
                self.fc4 = nn.Linear(50, output_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = self.fc4(x)
                return x
    else:

        class Net(nn.Module):

            def __init__(self, input_size, output_size):
                super().__init__()

                self.fc1 = nn.Linear(input_size, 50)
                self.fc2 = nn.Linear(50, 70)
                self.fc3 = nn.Linear(70, 50)
                self.fc4 = nn.Linear(50, 50)
                self.fc5 = nn.Linear(50, output_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)
                return x

    first_data = next(iter(loader))
    features, labels = first_data

    input_size = features.shape[-1]
    output_size = labels.shape[-1]
    output_size_corrs = 5

    from datetime import datetime

    weight_decay = 5
    curr_lr = 0.001
    EPOCHS = 240
    num_moms_corrs = 5

    now = datetime.now()
    lr_second = 0.99
    lr_first = 0.75
    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
    print('curr time: ', current_time)



    num_arrival_moms = 5

    import time

    net = Net(input_size, output_size).to(device)

    optimizer = optim.Adam(net.parameters(), lr=curr_lr,
                           weight_decay=(1 / 10 ** weight_decay))  # paramters is everything adjustable in model

    weight_decay = 5
    curr_lr = np.random.choice([.001, .0001])
    num_moms_corrs = 5
    now = datetime.now()
    lr_second = 0.99
    lr_first = 0.75
    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
    print('curr time: ', current_time)


    model_num = np.random.randint(0, 1000000)

    optimizer = optim.Adam(net.parameters(), lr=curr_lr,
                           weight_decay=(1 / 10 ** weight_decay))  # paramters is everything adjustable in model

    loss_list = []
    valid_list = []
    i = 0

    settings = (
        f"model_num_{model_num}_weight_decay_{weight_decay}_curr_lr_{curr_lr}_num_moms_corrs_{num_moms_corrs}_"
        f"nn_archi_{nn_archi}_max_lag_{max_lag}_max_power_1_{max_power_1}_max_power_2_{max_power_2}")

    print(settings)
    for epoch in range(EPOCHS):
        t_0 = time.time()
        for X, y in loader:
            i += 1

            if i % 10 == 12:
                all_errs = []
                for X1, y1 in loader_valid:
                    X1 = X1.float()
                    y1 = y1.float()
                    X1 = X1.to(device)
                    y1 = y1.to(device)
                    X1 = X1[:, 0, :]
                    y1 = y1[:, 0, :]
                    preds = net(X1)
                    curr_errs = 100 * torch.abs(
                        (torch.exp(preds[:, :]) - torch.exp(y1[:, :])) / torch.exp(y1[:, :])).mean(axis=0)
                    all_errs.append(curr_errs.reshape(1, -1))
                print(torch.vstack(all_errs).mean(axis=0))

            df_res = check_test(loader_valid, net)

            df_scv, df_rhos = scv_partion(df_res)
            if sys.platform == 'linux':
                dump_path = 'eliransc/scratch/MAP/results/scv_rho_df_res'+ settings+'.pkl'
                csv_file_scv = 'eliransc/scratch/MAP/results/scv_df_res'+ settings+'.pkl'
                csv_file_rho = 'eliransc/scratch/MAP/results/rho_df_res'+ settings+'.pkl'

            else:
                dump_path = r'C:\Users\Eshel\workspace\MAP\scv_rho_df_res'+ settings + '.pkl'
                csv_file_rho = r'C:\Users\Eshel\workspace\MAP\rho_df_res' + settings+ '.csv'
                csv_file_scv = r'C:\Users\Eshel\workspace\MAP\scv_df_res' + settings+   '.csv'

            df_scv.to_csv(csv_file_scv, index=False)
            df_rhos.to_csv(csv_file_rho, index=False)
            pkl.dump((df_scv, df_rhos), open(dump_path, 'wb'))

            # data_moms = Dataset_moms(features, labels, df, max_lag, max_power_1, max_power_2, num_arrival_moms, num_ser_moms)
            # X, y = data_moms.getitem()
            X = X.float()
            y = y.float()

            X = X.to(device)
            y = y.to(device)

            X = X[:, 0, :]
            y = y[:, 0, :]

            if torch.sum(torch.isinf(X)).item() == 0:

                net.zero_grad()
                output = net(X)
                loss = depart_loss(output, y)  # 1 of two major ways to calculate loss
                loss.backward()
                optimizer.step()
                net.zero_grad()
                # print(loss)
                if torch.isnan(loss).item():
                    break
            else:
                pass

        loss_list.append(loss.item())
        valid_list.append(valid(loader_valid, net).item())

        if len(loss_list) > 3:
            if check_loss_increasing(valid_list):
                curr_lr = curr_lr * lr_first
                optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                # print(curr_lr)
            else:
                curr_lr = curr_lr * lr_second
                optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                # print(curr_lr)

        print("Epoch: {}, Training: {:.5f}, Validation : {:.5f},  , Time: {:.3f}".format(epoch, loss.item(),
                                                                                         valid_list[-1],
                                                                                         time.time() - t_0))

        if sys.platform == 'linux':

            model_path = 'eliransc/scratch/MAP/models/moment_prediction'
        else:
            model_path = r'C:\Users\Eshel\workspace\MAP\models\moment_prediction'

        file_name_model = settings  +  '.pkl'

        torch.save(net.state_dict(), os.path.join(model_path, file_name_model))


if __name__ == '__main__':

    main()




